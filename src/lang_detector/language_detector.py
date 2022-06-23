from argparse import ArgumentParser
from typing import Iterator, Tuple, Any, List
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from pathlib import Path
from labels import LanguageEnum
from dataset import LanguageDataset
from tqdm import tqdm
import time
import torch
import pandas as pd

from lang_detector.model import LanguageDetectorModel
from lang_detector.utils import save_checkpoint, load_checkpoint


def parse_args():
    """Parse command line arguments."""
    p = ArgumentParser()
    p.add_argument('mode', choices=['train', 'eval'])
    p.add_argument('input_fn', type=str)
    p.add_argument('--output_fn', type=str, default='output.csv')
    p.add_argument('--log_interval', type=int, help='Log at every n-th step.',
                   default=500)
    p.add_argument('--embedding_size', type=int, default=64)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument('--lr', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=65)
    p.add_argument('--checkpoint_fn', type=str, help='Checkpoint filename',
                   default='my_checkpoint.pth.tar')
    p.add_argument('--restore_best_model', action="store_true",
                   help='If true, restores checkpoint given by checkpoint_fn')
    arguments = p.parse_args()
    return vars(arguments)


class DatasetPreprocessor:

    def __init__(self, path: str) -> None:

        self.path = path
        self.vocab = None
        self.tokenizer = None
        self.device = torch.device("cuda" if
                                   torch.cuda.is_available() else "cpu")

    def preprare_dataset(self) -> Tuple[LanguageDataset]:
        input_file = pd.read_csv(self.path)
        dataset = LanguageDataset(input_file,
                                  text_col="Text",
                                  label_col="Language")
        X = dataset.text
        y = dataset.labels
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)
        train_df = pd.DataFrame({'Text': X_train, 'Label': y_train})
        test_df = pd.DataFrame({'Text': X_test, 'Label': y_test})
        return (LanguageDataset(train_df.reset_index(drop=True),
                                text_col="Text", label_col="Label"),
                LanguageDataset(test_df.reset_index(drop=True),
                                text_col="Text", label_col="Label"))

    def build_vocab(self) -> None:
        train_dataset, test_dataset = self.preprare_dataset()
        train_iter = iter(train_dataset.dataset["Text"])
        vocab = build_vocab_from_iterator(yield_tokens(train_iter),
                                          specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        self.vocab = vocab

    def label_pipeline(self, x: int) -> int:
        # label_pipeline('10') --> '9' bcs zero index.
        return int(x - 1)

    def text_pipeline(self, x: str) -> List[int]:
        self.build_vocab()
        tokenizer = return_tokens
        return self.vocab(tokenizer(x))

    def collate_batch(self, batch: Tuple[Any]) ->  \
            Tuple[torch.Tensor]:
        """
        batch is a list of tuples with (example, label, length)
        where 'example' is a tensor of arbitrary shape
        and label/length are scalars

        Converts batch of data to tensor inputs for model.
        The offset is a tensor of delimiters to represent
        the beginning index of the individual sequence in the text tensor."""
        label_list, text_list, offsets = [], [], [0]
        for (_text, _label) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text),
                                          dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)


def yield_tokens(texts: Iterator) -> List[str]:
    tokenizer = get_tokenizer('basic_english')
    for text in texts:
        yield tokenizer(text)


def return_tokens(text: str) -> List[str]:
    tokenizer = get_tokenizer('basic_english')
    return tokenizer(text)


def train(train_dataloader: DataLoader,
          valid_dataloader: DataLoader,
          num_epochs: int,
          model: LanguageDetectorModel,
          device: torch.device,
          criterion,
          optimizer,
          scheduler) -> None:
    """Standard model training fucntion."""
    total_accu = None
    print('Beginning Training...')
    for epoch in range(1, num_epochs + 1):
        print(f'--- Epoch {epoch} ---')
        epoch_start_time = time.time()
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        for idx, (label, text, offsets) in tqdm(enumerate(train_dataloader),
                                                total=len(train_dataloader)):
            text = text.to(device)
            label = label.to(device)
            offsets = offsets.to(device)
            # Forward
            predicted = model(text, offsets)
            loss = criterion(predicted, label)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            # Optimizer step
            optimizer.step()

            total_acc += (predicted.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(epoch, idx,
                                                  len(train_dataloader),
                                                  total_acc/total_count))
                total_acc, total_count = 0, 0

        checkpoint = {"state_dict": model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)
        accu_val = evaluate(valid_dataloader, model, criterion)

        if total_accu is not None and total_accu > accu_val:
            # Learning rate scheduling should be applied
            # after optimizer’s update
            scheduler.step()
        else:
            total_accu = accu_val
            print('-' * 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                'valid accuracy {:8.3f} '.format(epoch,
                                                 time.time() - epoch_start_time,
                                                 accu_val))
            print('-' * 59)


def evaluate(dataloader, model, criterion) -> float:
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


def main():
    args = parse_args()
    preprocessor = DatasetPreprocessor(path=args['input_fn'])
    preprocessor.build_vocab()
    train_data = preprocessor.preprare_dataset()[0]
    test_data = preprocessor.preprare_dataset()[1]
    num_train = len(train_data)
    training_size = int(num_train * 0.95)
    split_train_, split_valid_ = random_split(train_data,
                                              [training_size, num_train - training_size])

    dataloader = DataLoader(train_data,
                            batch_size=8,
                            shuffle=False,
                            collate_fn=preprocessor.collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=args['batch_size'],
                                  shuffle=True,
                                  collate_fn=preprocessor.collate_batch)
    test_dataloader = DataLoader(test_data, batch_size=args['batch_size'],
                                 shuffle=True,
                                 collate_fn=preprocessor.collate_batch)

    num_class = len(set([lang.value for lang in LanguageEnum]))
    vocab_size = len(preprocessor.vocab)
    model = LanguageDetectorModel(vocab_size, args['embedding_size'],
                                  num_class).to(device=preprocessor.device)
    optim = torch.optim.SGD(model.parameters(), lr=args['lr'])

    if args['restore_best_model']:
        load_checkpoint(Path(__file__).parent.absolute() /
                        'checkpoints' / args['checkpoint_fn'],
                        model=model, optimizer=optim)
        print('Loaded state dicts.')

    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                                step_size=1,
                                                gamma=0.1)

    if args['mode'] == 'train':
        train(train_dataloader=dataloader,
              valid_dataloader=valid_dataloader,
              num_epochs=args['num_epochs'],
              model=model,
              device=preprocessor.device,
              criterion=criterion,
              optimizer=optim,
              scheduler=scheduler)

    if args['mode'] == 'eval':
        print('Checking the results of test dataset.')
        accu_test = evaluate(test_dataloader, model=model, criterion=criterion)
        print('test accuracy {:8.3f}'.format(accu_test))


if __name__ == "__main__":
    main()

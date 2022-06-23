import torch.nn as nn
import torch


class LanguageDetectorModel(nn.Module):
    """LanguageDetector model class with
    embedding and linear layers."""

    def __init__(self, vocab_size, embed_dim, num_class) -> None:
        super(LanguageDetectorModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, x, offset) -> torch.Tensor:
        embedded = self.embedding(x, offset)
        return self.fc(embedded)

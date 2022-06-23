from torch.utils.data import Dataset
from pandas.core.frame import DataFrame
from typing import Iterable, Any

from lang_detector.labels import LanguageEnum


class LanguageDataset(Dataset):
    """Dataset class for language detection."""

    def __init__(self,
                 dataset: DataFrame,
                 text_col: str,
                 label_col: str,
                 transform=None) -> None:

        self.dataset = dataset
        self.label_names = self.dataset[label_col]
        self.labels = self.map_label_to_id()
        self.text = self.dataset[text_col]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.dataset.iloc[idx]
    
    def map_label_to_id(self):
        if checktype(self.label_names, str):
            return [LanguageEnum[lang].value for lang in self.label_names]
        else:
            return self.label_names


def checktype(iterable: Iterable[Any], type: Any) -> bool:
    return bool(all(isinstance(elem, type) for elem in iterable))

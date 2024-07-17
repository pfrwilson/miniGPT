import random
from torch.utils.data import Dataset
from dataclasses import dataclass
import logging
from utils.registry import Registry
from abc import ABC, abstractmethod

logger = logging.getLogger("DatasetFactory")


class TextFileDataset(Dataset):
    def __init__(self, textfile, chunk_length):
        self.textfile = textfile
        self.chunk_length = chunk_length

        with open(textfile, "r") as f:
            self.text = f.read()

        self._n_chunks = len(self.text) // self.chunk_length

        self.vocabulary = list(set(self.text))
        self.idx2char = self.vocabulary
        self.char2idx = {v: i for i, v in enumerate(self.vocabulary)}

    def _get_chunk(self, chunk_idx):
        return self.text[
            chunk_idx * self.chunk_length : chunk_idx * self.chunk_length
            + self.chunk_length
        ]

    def __len__(self):
        return self._n_chunks

    def __getitem__(self, idx):
        if idx < 0 or idx >= self._n_chunks:
            raise IndexError

        raw_text = self._get_chunk(idx)
        return {"text": raw_text}


class OpenWebTextWrapper(Dataset): 
    def __init__(self, dataset, chunk_length=500): 
        self.dataset = dataset 
        self.chunk_length = chunk_length
    
    def __len__(self): 
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        if len(text) > self.chunk_length: 
            start_idx = random.sample(
                range(0, len(text) - self.chunk_length), 1
            )[0]
            text = text[start_idx:start_idx + self.chunk_length]
        item['text'] = text 
        return item


dataset_factories = Registry()


@dataclass
class BaseDatasetFactory:
    @abstractmethod
    def __call__(self): ...


@dataset_factories.register("openwebtext")
@dataclass
class OpenWebText(BaseDatasetFactory):
    item_limit: int | None = None
    keep_in_memory: bool = False
    chunk_length: int = 500

    def __call__(self):
        logger.info(f"Instantiating OpenWebText... this could take a few seconds!")
        from datasets import load_from_disk

        dataset = load_from_disk("/datasets/openwebtext", keep_in_memory=self.keep_in_memory)
        dataset = dataset["train"]
        dataset = OpenWebTextWrapper(dataset, chunk_length=self.chunk_length)

        if self.item_limit is not None:
            from torch.utils.data import Subset

            dataset = Subset(dataset, list(range(self.item_limit)))
        return dataset


@dataset_factories.register("textfile_dataset")
@dataclass
class TextFileDatasetFactory(BaseDatasetFactory):
    path: str = "data/bible.txt"
    chunk_length: int = 500

    def __call__(self):
        return TextFileDataset(self.path, chunk_length=self.chunk_length)

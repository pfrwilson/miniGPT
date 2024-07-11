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


dataset_factories = Registry()


@dataclass
class BaseDatasetFactory:
    @abstractmethod 
    def __call__(self): 
        ...


@dataset_factories.register('openwebtext')
@dataclass
class OpenWebText(BaseDatasetFactory): 
    item_limit: int | None = None 

    def __call__(self): 
        logger.info(f"Instantiating OpenWebText... this could take a few seconds!")
        from datasets import load_from_disk
        dataset = load_from_disk('/datasets/openwebtext')
        dataset = dataset['train']
        if self.item_limit is not None: 
            from torch.utils.data import Subset
            dataset = Subset(dataset, list(range(self.item_limit)))
        return dataset

@dataset_factories.register('textfile_dataset')
@dataclass
class TextFileDatasetFactory(BaseDatasetFactory):
    path: str = 'data/bible.txt'
    chunk_length: int = 500

    def __call__(self):
        return TextFileDataset(self.path, chunk_length=self.chunk_length)
    
        

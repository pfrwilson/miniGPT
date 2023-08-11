from torch.utils.data import Dataset
from dataclasses import dataclass
import logging


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


from torchzero.utils.registry import Registry
dataset_registry = Registry()


@dataset_registry.register_factory
class openwebtext():
    @dataclass
    class Config: 
        name = 'openwebtext'
        item_limit: int | None = None, # Optionally specify a limit to the number of items included in the dataset!

    def __call__(self, cfg):
        logger.info(f"Instantiating OpenWebText... this could take a few seconds!")
        from datasets import load_from_disk
        dataset = load_from_disk('/datasets/openwebtext')
        dataset = dataset['train']
        if cfg.item_limit is not None: 
            from torch.utils.data import Subset
            dataset = Subset(dataset, list(range(cfg.item_limit)))
        return dataset

@dataset_registry.register_factory
class Bible:
    @dataclass
    class Config: 
        name = 'bible'
        chunk_length: int = 500

    def __call__(self, config: Config):
        return TextFileDataset('data/bible.txt', chunk_length=config.chunk_length)
    

# if __name__ == "__main__": 
#     print(DATASETS)
#     print(DATASET_CONFIGS)
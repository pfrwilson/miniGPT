"""
Train a model to generate text.

Author: Paul Wilson 2023
"""


import numpy as np
import torch
import torch.nn as nn
from torch.nn import RNN
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb
from dataclasses import dataclass, field
import coolname
from os.path import join
import os
import typing as t
import time
import yaml
import json
from datetime import datetime
from models import LSTMClassifier
import sys
from argparse import ArgumentParser
from torchzero.nn import LSTMClassifier
from torch import nn
import torch
from torch.nn import functional as F
from torchzero.utils.tokenizer import Tokenizer
import einops
import rich 
import logging
import simple_parsing
from simple_parsing.helpers import JsonSerializable
from dataset import dataset_registry
from trainable_models import registry as model_registry, TrainableModel
# breakpoint()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_batch(batch, tokenizer, start_token=True, max_len=None, pad='right', truncate='right'):
    """
    Prepares a batch of data for input into a neural network model. 
    Args: 
        batch (dict) - A batch of data. Should be a dictionary with at least the key "text", 
        and batch["text"] should be a list of lists of strings. 
    """

    text = batch["text"]
    encodings = [tokenizer.encode(item) for item in text]
    lengths = [len(encoding) for encoding in encodings]
    largest_len = max(lengths)

    # pad to max length
    padding_token = tokenizer.vocab2idx["<PAD>"]
    encodings_padded = []
    for encoding in encodings:
        if pad.lower() == "left":
            encoding = encoding + [padding_token] * (largest_len - len(encoding))
        elif pad.lower() == "right": 
            encoding = [padding_token] * (largest_len - len(encoding)) + encoding
        encodings_padded.append(encoding)

    if start_token:
        # add start token for input (not necessarily needed for rnn models, but nice.)
        start_token = tokenizer.vocab2idx["<START>"]
        for encoding in encodings_padded:
            encoding.insert(0, start_token)

    X = torch.tensor(encodings_padded)

    if max_len is not None and X.shape[1] > max_len:
        if truncate == 'right':    
            X = X[:, :max_len]
        elif truncate == 'left': 
            X = X[:, -max_len:]

    return X


class BibleCharacterChunksDataset:
    def __init__(self, path, chunk_length=100, split="train"):
        self.path = path
        with open(path, "r") as f:
            self._raw_data = f.read()

        import re

        match = re.search(
            "\*\*\* START OF THE PROJECT GUTENBERG EBOOK THE KING JAMES BIBLE \*\*\*",
            self._raw_data,
        )
        start_bible = match.span()[1]
        match = re.search(
            "\*\*\* END OF THE PROJECT GUTENBERG EBOOK THE KING JAMES BIBLE \*\*\*",
            self._raw_data,
        )
        end_bible = match.span()[0]

        self._bible = self._raw_data[start_bible:end_bible]

        self._chunk_length = chunk_length

        self.vocab = sorted(list(set(self._bible)))
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}

        self._train = self._bible[: int(0.8 * len(self._bible))]
        self._test = self._bible[int(0.8 * len(self._bible)) :]

        self.split = split
        self._bible = self._train if split == "train" else self._test

    def __len__(self):
        return len(self._bible) // self._chunk_length

    def __getitem__(self, idx):
        start = idx * self._chunk_length
        end = (idx + 1) * self._chunk_length

        offset = 0
        if self.split == "train" and idx != 0 and idx != len(self) - 1:
            # add a random offset to the start of the chunk
            offset = np.random.randint(0, self._chunk_length)

        start += offset
        end += offset

        text = self._bible[idx * self._chunk_length : (idx + 1) * self._chunk_length]
        return np.array(self.encode(text))

    def decode(self, idxs):
        if isinstance(idxs, torch.Tensor):
            idxs = idxs.tolist()
        return "".join([self.idx2char[i] for i in idxs])

    def encode(self, text):
        return [self.char2idx[c] for c in text]


@dataclass
class Config:
    lr: float = 0.001
    batch_size: int = 256
    n_epochs: int = 100
    model: model_registry.BaseConfig = simple_parsing.subgroups(
        {
            name: model_registry.get_config(name) for name in model_registry.list_constructibles()     
        }, default='premade'
    )
    # dataset_name: 

    name: str = field(
        default_factory=lambda: f"{datetime.now().strftime('%Y-%m-%d')}_{coolname.generate_slug(2)}"
    )

    model_path: str | None = None  # optional path to a model to load
    vocab_path: str = "data/vocab_1024.json"  # optional path to a vocab to load
    wandb: bool = True
    debug: bool = False


    dataset: dataset_registry.BaseConfig = simple_parsing.subgroups(
        {name: dataset_registry.get_config(name) for name in dataset_registry.list_constructibles()}, 
        default=dataset_registry.list_constructibles()[0])

    def __post_init__(self):
        self.exp_dir = join("logs", self.name)


def train(args: Config):
    os.makedirs(args.exp_dir, exist_ok=True)
    
    if args.wandb and not args.debug:
        wandb.init(
            project="auto-preacher", config=vars(args), name=os.path.basename(args.exp_dir)
        )
    else: 
        wandb.log = lambda x: rich.print(x)

    if args.debug: 
        logging.basicConfig(
            level=logging.DEBUG
        )

    # Dataset
    train_loader, test_loader = create_dataloaders(args)

    # Tokenizer
    from torchzero.utils.tokenizer import Tokenizer
    tokenizer = Tokenizer.from_json(args.vocab_path)

    # Model
    model: TrainableModel = model_registry.build(args.model, tokenizer=tokenizer).to(DEVICE)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

    # Optimizer
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=5, verbose=True
    )
    best_val_loss = np.inf

    for epoch in range(1, args.n_epochs):
        train_loss = train_epoch(train_loader, model, tokenizer, opt)
        val_loss = val_epoch(test_loader, model, tokenizer)

        scheduler.step(val_loss)

        print(f"===== EPOCH {epoch} ======")
        print(f"LOSSES: ")
        print(f"TRAIN: {train_loss}")
        print(f"VAL: {val_loss}")
        print(f"GENERATED TEXT: ")
        print(model.predict(prompt=None, max_len=500))
        print()
        print("=========================")

        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch,
                "lr": opt.param_groups[0]["lr"],
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), join(args.exp_dir, "best_model.pt"))

            if args.wandb:
                wandb.run.summary["best_val_loss"] = best_val_loss

            print("Saved model!")


def create_dataloaders(args: Config):
    # from dataset import TextFileDataset
    import torch

    #dataset = TextFileDataset("data/bible.txt", chunk_length=args.chunk_length)
    dataset = dataset_registry.build(args.dataset)
    from torch.utils.data import Subset

    train_dataset = Subset(dataset, range(int(len(dataset) * 0.8)))
    test_dataset = Subset(dataset, range(int(len(dataset) * 0.8), len(dataset)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    return train_loader, test_loader


def train_epoch(loader, model: TrainableModel, tokenizer, optimizer):
    loss_epoch = 0
    total = 0
    model.train()

    for iteration, batch in enumerate(tqdm(loader, leave=False)):
        loss = model.step(batch)
        total += 1

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_epoch += loss.item()

    return loss_epoch / total


@torch.no_grad()
def val_epoch(loader, model: TrainableModel, tokenizer):
    loss_epoch = 0
    total = 0
    model.eval()

    for iteration, batch in enumerate(tqdm(loader, leave=False)):
        loss = model.step(batch)
        total += 1

        loss_epoch += loss.item()

    return loss_epoch / total


def parse_args(args=None):
    return simple_parsing.parse(Config, args=args)


if __name__ == "__main__":
    import sys

    args = parse_args()
    train(args)


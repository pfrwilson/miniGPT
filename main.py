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
from datetime import datetime
from models import LSTMClassifier


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def step(model, batch):
    X = batch.to(DEVICE)
    state = {}

    loss = torch.tensor(0.0, device=DEVICE)
    for i in range(X.shape[1] - 1):
        current_char = X[:, i]
        target = X[:, i + 1]

        y, state = model(current_char, **state)

        loss_step = F.cross_entropy(y, target)
        loss += loss_step

    return loss


def train_epoch(loader, model, optimizer):
    loss_epoch = 0
    total = 0

    for iteration, batch in enumerate(tqdm(loader, leave=False)):
        loss = step(model, batch)
        total += 1

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_epoch += loss.item()

    return loss_epoch / total


@torch.no_grad()
def val_epoch(loader, model):
    loss_epoch = 0
    total = 0

    for iteration, batch in enumerate(tqdm(loader, leave=False)):
        loss = step(model, batch)
        total += 1

        loss_epoch += loss.item()

    return loss_epoch / total


def predict(dataset, model, len=100):
    state = {}
    import random

    seed = random.sample(dataset.vocab, 1)[0]
    X = torch.tensor(dataset.encode([seed])).unsqueeze(0)
    X = X.to(DEVICE)

    for i in range(len):
        current_char = X[:, 0]

        y, state = model(current_char, **state)

        # randomly sample from the distribution
        y = F.softmax(y, dim=-1)
        y = torch.multinomial(y, 1)
        X = y

        print(dataset.decode([y.item()]), end="")


def predict_live(dataset, model):
    state = {}
    import random

    seed = random.sample(dataset.vocab, 1)[0]
    X = torch.tensor(dataset.encode([seed])).unsqueeze(0)
    X = X.to(DEVICE)

    while True:
        current_char = X[:, 0]

        y, state = model(current_char, **state)

        # randomly sample from the distribution
        y = F.softmax(y, dim=-1)
        y = torch.multinomial(y, 1)
        X = y

        print(dataset.decode([y.item()]), end="", flush=True)
        time.sleep(0.1)


@dataclass
class Config:
    mode: t.Literal["train", "predict"] = "train"

    lr: float = 0.001
    batch_size: int = 256
    n_epochs: int = 100
    n_features: int = 128
    n_layers: int = 1  # number of layers in the RNN
    residual: bool = False  # whether to use residual connections between layers
    chunk_length: int = 100

    name: str = field(
        default_factory=lambda: f"{datetime.now().strftime('%Y-%m-%d')}_{coolname.generate_slug(2)}"
    )

    model_path: str | None = None  # optional path to a model to load

    def __post_init__(self):
        self.exp_dir = join("logs", self.name)
        self.command = " ".join(sys.argv)


def train(args: Config):
    os.makedirs(args.exp_dir, exist_ok=True)
    # easiest way to save the config is just to save the command line to a file
    with open(join(args.exp_dir, "command.txt"), "w") as f:
        f.write(args.command)

    import wandb

    wandb.init(
        project="auto-preacher", config=vars(args), name=os.path.basename(args.exp_dir)
    )

    train_dataset = BibleCharacterChunksDataset("data/bible.txt")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataset = BibleCharacterChunksDataset("data/bible.txt", split="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = LSTMClassifier(
        vocab_size=len(train_dataset.vocab),
        n_features=args.n_features,
        n_classes=len(train_dataset.vocab),
        n_layers=args.n_layers,
        residual=args.residual,
    ).to(DEVICE)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", patience=5, verbose=True
    )

    best_val_loss = np.inf

    for epoch in range(1, args.n_epochs):
        train_loss = train_epoch(train_loader, model, opt)
        val_loss = val_epoch(test_loader, model)

        scheduler.step(val_loss)

        print(f"===== EPOCH {epoch} ======")
        print(f"LOSSES: ")
        print(f"TRAIN: {train_loss}")
        print(f"VAL: {val_loss}")
        print(f"GENERATED TEXT: ")
        predict(train_dataset, model, 500)
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

            wandb.run.summary["best_val_loss"] = best_val_loss

            print("Saved model!")


def predict(args: Config):
    train_dataset = BibleCharacterChunksDataset("data/bible.txt")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataset = BibleCharacterChunksDataset("data/bible.txt", split="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = LSTMClassifier(
        vocab_size=len(train_dataset.vocab),
        n_features=args.n_features,
        n_classes=len(train_dataset.vocab),
        n_layers=args.n_layers,
        residual=args.residual,
    ).to(DEVICE)

    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

    predict_live(train_dataset, model)


if __name__ == "__main__":
    import simple_parsing
    import sys

    args = simple_parsing.parse(Config)

    if args.mode == "train":
        train(args)
    elif args.mode == "predict":
        predict(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

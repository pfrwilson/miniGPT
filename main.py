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


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BibleCharacterChunksDataset: 
    def __init__(self, path, chunk_length=100, split='train'): 
        self.path = path 
        with open(path, 'r') as f: 
            self._raw_data = f.read()

        import re 
        match = re.search("\*\*\* START OF THE PROJECT GUTENBERG EBOOK THE KING JAMES BIBLE \*\*\*", self._raw_data)
        start_bible = match.span()[1]
        match = re.search("\*\*\* END OF THE PROJECT GUTENBERG EBOOK THE KING JAMES BIBLE \*\*\*", self._raw_data)
        end_bible = match.span()[0]

        self._bible = self._raw_data[start_bible:end_bible]

        self._chunk_length = chunk_length

        self.vocab = sorted(list(set(self._bible)))
        self.char2idx = {c:i for i, c in enumerate(self.vocab)}
        self.idx2char = {i:c for i, c in enumerate(self.vocab)}

        self._train = self._bible[:int(0.8*len(self._bible))]
        self._test = self._bible[int(0.8*len(self._bible)):]

        self.split = split
        self._bible = self._train if split == 'train' else self._test

    def __len__(self):
        return len(self._bible) // self._chunk_length
    
    def __getitem__(self, idx):
        start = idx*self._chunk_length
        end = (idx+1)*self._chunk_length

        offset = 0
        if self.split == 'train' and idx != 0 and idx != len(self) - 1: 
            # add a random offset to the start of the chunk
            offset = np.random.randint(0, self._chunk_length)
    
        start += offset
        end += offset

        text = self._bible[idx*self._chunk_length:(idx+1)*self._chunk_length]
        return np.array(self.encode(text))
    
    def decode(self, idxs):
        if isinstance(idxs, torch.Tensor): 
            idxs = idxs.tolist()
        return ''.join([self.idx2char[i] for i in idxs])

    def encode(self, text): 
        return [self.char2idx[c] for c in text]


class VanillaRNNCell(nn.Module): 
    def __init__(self, n_features): 
        super().__init__()
        self.n_features = n_features
        self.layer = nn.Linear(n_features * 2, n_features)
        self.activation = nn.Tanh()

    def forward(self, X, h=None): 
        if h is None: 
            h = torch.zeros(X.shape[0], self.n_features, device=X.device)

        X = torch.cat([X, h], dim=-1)

        return self.activation(self.layer(X) + h)
    

class LSTMCell(nn.Module):
    def __init__(self, n_features): 
        super().__init__()
        self.n_features = n_features 

        self.forget_layer = torch.nn.Linear(n_features * 2, n_features)
        self.input_layer = torch.nn.Linear(n_features * 2, n_features)
        self.input_gate_layer = torch.nn.Linear(n_features * 2, n_features)
        self.output_layer = torch.nn.Linear(n_features, n_features)
        self.output_gate_layer = torch.nn.Linear(n_features * 2, n_features)

    def forward(self, X, h=None, c=None): 
        if h is None: 
            h = torch.zeros(X.shape[0], self.n_features, device=X.device)
        if c is None: 
            c = torch.zeros(X.shape[0], self.n_features, device=X.device)

        X = torch.cat([X, h], dim=-1)

        forget_gate = torch.sigmoid(self.forget_layer(X))
        input_gate = torch.sigmoid(self.input_gate_layer(X))
        output_gate = torch.sigmoid(self.output_gate_layer(X))

        input_ = torch.tanh(self.input_layer(X))
        
        input_ = input_gate * input_
        c  = forget_gate * c + input_

        h = output_gate * torch.tanh(c)

        return h, c
    

class RNNClassifier(nn.Module): 
    def __init__(self, vocab_size, n_features, n_classes, cell_type='vanilla'):
        self.vocab_size = vocab_size
        self.n_features = n_features
        self.n_classes = n_classes
        self.cell_type = cell_type

        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, n_features)
        if cell_type == 'vanilla':
            self.cell = VanillaRNNCell(
                n_features=n_features
            )
        elif cell_type == 'lstm':
            self.cell = LSTMCell(
                n_features=n_features
            )
        else:
            raise ValueError(f"Unknown cell type {cell_type}")
        
        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, X, **kwargs): 
        """
        X should be a single token (shape batch_size,))
        """
        if self.cell_type == 'vanilla': 
            h = kwargs['h'] if 'h' in kwargs else None
        if self.cell_type == 'lstm': 
            h = kwargs['h'] if 'h' in kwargs else None
            c = kwargs['c'] if 'c' in kwargs else None

        X = self.embeddings(X)
        if self.cell_type == 'vanilla':
            h = self.cell(X, h)
            outs = {'h': h}
        elif self.cell_type == 'lstm':
            h, c = self.cell(X, h, c)
            outs = {'h': h, 'c': c}
        else:
            raise ValueError(f"Unknown cell type {self.cell_type}")
        
        y = self.classifier(h)

        return y, outs 
        

def step(model, batch): 
    X = batch.to(DEVICE)
    state = {}

    loss = torch.tensor(0.0, device=DEVICE)
    for i in range(X.shape[1] - 1):

        current_char = X[:, i]
        target = X[:, i+1]

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

        print(dataset.decode([y.item()]), end='')


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

        print(dataset.decode([y.item()]), end='', flush=True)
        time.sleep(0.)


@dataclass
class Config:
    mode: t.Literal['train', 'predict'] = 'train'

    cell_type: str = 'lstm'
    lr: float = 0.001
    batch_size: int = 256
    n_epochs: int = 100
    n_features: int = 128
    chunk_length: int = 100

    exp_dir: str = field(default_factory=lambda: join('logs', coolname.generate_slug(2)))

    model_path: str | None = None # optional path to a model to load


def main(args: Config): 

    os.makedirs(args.exp_dir, exist_ok=True)

    import wandb
    wandb.init(project="auto-preacher", config=vars(args), name=os.path.basename(args.exp_dir))

    train_dataset = BibleCharacterChunksDataset('data/bible.txt')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = BibleCharacterChunksDataset('data/bible.txt', split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = RNNClassifier(
        vocab_size=len(train_dataset.vocab),
        n_features=args.n_features,
        n_classes=len(train_dataset.vocab),
        cell_type=args.cell_type
    ).to(DEVICE)

    if args.model_path: 
        model.load_state_dict(torch.load(args.model_path))

    if args.mode == 'train': 
        opt = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, verbose=True)

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

            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': epoch,
                'lr': opt.param_groups[0]['lr']
            })

            if val_loss < best_val_loss: 
                best_val_loss = val_loss
                torch.save(model.state_dict(), join(args.exp_dir,'best_model.pt'))

                wandb.run.summary["best_val_loss"] = best_val_loss

                print("Saved model!")

    elif args.mode == 'predict':
        predict_live(train_dataset, model)


if __name__ == "__main__": 
    import simple_parsing
    from simple_parsing import parse_known_args

    #args, unknown_ = parse_known_args()

    args = simple_parsing.parse(Config)
    main(args)
    

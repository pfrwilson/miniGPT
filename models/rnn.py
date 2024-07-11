"""
Implementations of various recurrent neural network components. 

Author: Paul Wilson

========== Sources ================ 

LSTM: 
@article{graves2012long,
  title={Long short-term memory},
  author={Graves, Alex and Graves, Alex},
  journal={Supervised sequence labelling with recurrent neural networks},
  pages={37--45},
  year={2012},
  publisher={Springer}
}

RNN:
@article{medsker2001recurrent,
  title={Recurrent neural networks},
  author={Medsker, Larry R and Jain, LC},
  journal={Design and Applications},
  volume={5},
  number={64-67},
  pages={2},
  year={2001}
}
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import RNN
import torch.nn.functional as F
import torch.optim as optim


class VanillaRNNCell(nn.Module):
    """Basic RNN Cell with input to state and state to state components"""
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.layer = nn.Linear(n_features * 2, n_features)
        self.activation = nn.Tanh()

    def forward(self, X, h=None):
        if h is None:
            h = torch.zeros(X.shape[0], self.n_features, device=X.device)

        X = torch.cat([X, h], dim=-1)

        h = self.activation(self.layer(X) + h)
        return {"h": h}


class LSTMCell(nn.Module):
    """
    LSTM Cell with forget gate, input gate, output gate, etc.
    This class is just for demonstration purposes, and is not used in the final model.
    The class below "FastLSTMCell" is used instead.
    """
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

        self.W_f = torch.nn.Linear(n_features * 2, n_features)
        self.W_i = torch.nn.Linear(n_features * 2, n_features)
        self.W_g = torch.nn.Linear(n_features * 2, n_features)
        self.W_o = torch.nn.Linear(n_features * 2, n_features)

    def forward(
        self,
        X,
        h=None,
        c=None,
    ):
        if h is None:
            h = torch.zeros(X.shape[0], self.n_features, device=X.device)
        if c is None:
            c = torch.zeros(X.shape[0], self.n_features, device=X.device)

        X = torch.cat([X, h], dim=-1)

        f = torch.sigmoid(self.W_f(X))
        i = torch.sigmoid(self.W_i(X))
        o = torch.sigmoid(self.W_o(X))
        g = torch.tanh(self.W_g(X))

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c


class FastLSTMCell(nn.Module):
    """
    An lstm cell which is more efficient than the one above because it combines
    state-to-state and input-to-state operations into a single matrix multiplication.
    """
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

        self.W = torch.nn.Linear(n_features * 2, n_features * 4)

    def forward(
        self,
        X,
        h=None,
        c=None,
    ):
        if h is None:
            h = torch.zeros(X.shape[0], self.n_features, device=X.device)
        if c is None:
            c = torch.zeros(X.shape[0], self.n_features, device=X.device)

        X = torch.cat([X, h], dim=-1)

        fiog = self.W(X)

        f, i, o, g = torch.split(fiog, self.n_features, dim=-1)
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, n_features, n_classes, n_layers=1, residual=False):
        self.vocab_size = vocab_size
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.residual = residual

        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, n_features)
        self.cells = nn.ModuleList([FastLSTMCell(n_features) for _ in range(n_layers)])
        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x, h=None, c=None):
        x = self.embeddings(x)

        h_last = h or torch.zeros(self.n_layers, x.shape[0], self.n_features, device=x.device)
        c_last = c or torch.zeros(self.n_layers, x.shape[0], self.n_features, device=x.device)
        h_new = []
        c_new = []

        for i, layer in enumerate(self.cells):
            h, c = layer(x, h_last[i], c_last[i])
            h_new.append(h)
            c_new.append(c)

            if self.residual:
                x = x + h
            else:
                x = h

        y = self.classifier(x)

        return y, {"h": h_new, "c": c_new}


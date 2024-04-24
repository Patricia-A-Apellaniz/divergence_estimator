# Author: Ana Jiménez & Patricia A. Apellániz
# Email: patricia.alonsod@upm.es
# Date: 15/04/2024


# Packages to import
import copy
import torch

from torch import nn, optim
from torch.nn import functional as F

import warnings

warnings.filterwarnings('ignore')

_activations = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(0.2),
    'tanh': nn.Tanh(),
}


class DenseModule(nn.Module):
    def __init__(self, n_neurons: int, activation: str, *args, batch_norm: bool, dropout: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer = nn.LazyLinear(out_features=n_neurons)
        if activation not in _activations.keys():
            msg = f'Expected one of {_activations.keys()}'
            raise ValueError(msg)
        self.activation = _activations[activation]
        self.batch_norm = None
        if batch_norm:
            self.batch_norm = nn.LazyBatchNorm1d()
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.layer(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers_ = []
        for elem in layers:
            layers_.append(DenseModule(elem, activation='leaky_relu', batch_norm=True, dropout=True))
        layers_ += [nn.LazyLinear(1)]
        self.l = nn.ModuleList(layers_)

    def forward(self, data, sigmoid=False):
        x = data
        for layer in self.l:
            x = layer(x)
        if sigmoid:
            x = F.sigmoid(x)
        return x.reshape(-1)

    @torch.no_grad()
    def predict(self, data, sigmoid=True):
        self.train(False)
        _X_y = 2
        out = []
        for batch in data:
            if len(batch) == _X_y:
                x, _ = batch
            else:
                x = batch
            y = self.forward(x, sigmoid=sigmoid)
            out.append(y)
        return torch.cat(out)

    def train_loop(self, train_dl, val_dl, epochs, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr)
        tr_loss = []
        eval_loss = []
        patience_0 = 1000  # Number of epochs to wait before stopping
        patience = patience_0
        best_metric = float('inf')  # For loss, set it to float('inf'); for accuracy, set it to 0
        best_model = None
        for epoch in range(epochs):
            self.train(True)
            cum_loss = 0.0
            cum_eval_loss = 0.0
            for X, y in train_dl:
                optimizer.zero_grad()
                logit_x = self(X)
                loss = F.binary_cross_entropy_with_logits(logit_x, y.reshape(-1))
                loss.backward()
                optimizer.step()
                cum_loss += loss.item()

            avg_loss = cum_loss / len(train_dl)
            tr_loss.append(avg_loss * 2)

            # Evaluation for early stopping
            self.eval()
            with torch.no_grad():
                for X_eval, y_eval in val_dl:
                    logit_x_eval = self(X_eval)
                    loss_eval = F.binary_cross_entropy_with_logits(logit_x_eval, y_eval.reshape(-1))
                    cum_eval_loss += loss_eval.item()
                avg_loss_eval = cum_eval_loss / (len(val_dl))
                eval_loss.append(avg_loss_eval * 2)

            # Check if the validation loss (or accuracy) has improved
            if avg_loss_eval < best_metric:
                best_metric = avg_loss_eval
                patience = patience_0  # Reset patience
                best_model = copy.deepcopy(self.state_dict())
            else:
                patience -= 1

            if patience == 0:
                # print('Early stopping, no improvement in validation loss.')
                self.load_state_dict(best_model)
                break

        return tr_loss, eval_loss

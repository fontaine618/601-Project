import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABCMeta
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


def strikezone_dict_to_array(szd):
    levels = []
    strikezones = []
    for lvls, sz in szd.items():
        levels.append(lvls)
        strikezones.append(sz.ravel())
    strikezones = np.array(strikezones)
    return levels, strikezones


def array_to_strikezone_dict(levels, arr):
    size = int(np.sqrt(arr.shape[1]))
    return {
        lvls: arr[i, :].reshape((size, size))
        for i, lvls in enumerate(levels)
    }


class TensorDataset(Dataset):
    """Custom Dataset for loading"""

    def __init__(self, X):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        """
        self.X = torch.tensor(X).float()

    def __getitem__(self, item):
        return self.X[item, :]

    def __len__(self):
        return self.X.size(0)


class Autoencoder(TransformerMixin, BaseEstimator, metaclass=ABCMeta):

    def __init__(self, model=None, loss=F.mse_loss,
                 optimizer=None):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def fit(self, X, y=None, **fit_params):
        """Fit the model with X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : None
            Ignored variable.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X, **fit_params)
        return self

    def _fit(self, X, num_epochs=50, batch_size=None):
        """Dispatch to the right submethod depending on the chosen solver."""
        X = check_array(X, dtype=[np.float64, np.float32], ensure_2d=True)
        if batch_size is None:
            batch_size = X.shape[0]
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []
        for epoch in range(num_epochs):
            for X in dataloader:
                X = X.cuda()
                # forward step
                out = self.model(X)
                loss = self.loss(out, X)
                losses.append(loss.data.item())
                # backward step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print('epoch [{}/{}], loss:{:.4f}'.format(
                epoch + 1, num_epochs, loss.data.item()
            ))
        self.losses_ = np.array(losses)
        self.fitted_ = True

    def transform(self, X):
        """Apply dimensionality reduction to X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self)
        X = check_array(X)
        with torch.no_grad():
            X = torch.tensor(X).cuda().float()
            return self.model.encoder(X).cpu().numpy()

    def inverse_transform(self, U):
        """Transform data back to its original space.
        In other words, return an input X whose transform would be U.
        Parameters
        ----------
        U : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.
        Returns
        -------
        X array-like, shape (n_samples, n_features)
        """
        with torch.no_grad():
            U = torch.tensor(U).cuda().float()
            return self.model.decoder(U).cpu().numpy().clip(0, 1)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model with X and apply the dimensionality reduction on X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : None
            Ignored variable.
        Returns
        -------
        U : array-like, shape (n_samples, n_components)
            Transformed values.
        """
        self._fit(X, **fit_params)
        return self.transform(X)

    def score_samples(self, X):
        """Return the MSE of each sample
        Parameters
        ----------
        X : array, shape(n_samples, n_features)
            The data.
        Returns
        -------
        mse : array, shape (n_samples,)
            MSE of each sample under the current model.
        """
        check_is_fitted(self)
        X = check_array(X)
        U = self.transform(X)
        return mean_squared_error(X.T, self.inverse_transform(U).T, multioutput='raw_values').T

    def score(self, X, y=None):
        """Return the average MSE of all samples.
        Parameters
        ----------
        X : array, shape(n_samples, n_features)
            The data.
        y : None
            Ignored variable.
        Returns
        -------
        mse : float
            Average MSE of the samples under the current model.
        """
        return np.mean(self.score_samples(X))


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Logit(nn.Module):
    def __init__(self):
        super(Logit, self).__init__()

    def forward(self, x):
        return torch.log(torch.div(x, torch.add(1., torch.mul(x, -1.))))


class NNAutoencoder(nn.Module):

    def __init__(
            self,
            size=100**2,
            n_components=5,
            layers=[1024, 128]
    ):
        super(NNAutoencoder, self).__init__()
        self.n_components = n_components
        encoding_layers = []
        prev = size
        for units in layers:
            encoding_layers.append(nn.Linear(prev, units))
            encoding_layers.append(nn.Tanh())
            prev = units
        encoding_layers.append(nn.Linear(prev, self.n_components))
        encoding_layers.append(nn.Tanh())
        self.encoder = nn.Sequential(*encoding_layers)
        decoding_layers = []
        prev = self.n_components
        for units in layers[::-1]:
            decoding_layers.append(nn.Linear(prev, units))
            decoding_layers.append(nn.Tanh())
            prev = units
        decoding_layers.append(nn.Linear(prev, size))
        decoding_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoding_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CNNAutoencoder(nn.Module):

    def __init__(
            self,
            size=100,
            n_components=5
    ):
        super(CNNAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            View((-1, 1, size, size)),
            nn.Conv2d(1, 4, 69, stride=1, padding=0),
            nn.AvgPool2d(2, stride=2),
            #nn.Sigmoid(),
            #nn.ReLU(True),
            nn.Conv2d(4, 16, 9, stride=1, padding=0),
            nn.AvgPool2d(2, stride=2),
            #nn.Sigmoid(),
            #nn.ReLU(True),
            nn.Conv2d(16, 32, 4, stride=1, padding=0),
            #nn.Sigmoid(),
            #nn.ReLU(True),
            View((-1, 32)),
            #nn.Linear(32, 32),
            #nn.ReLU(True),
            nn.Linear(32, n_components),
            #nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_components, 32),
            #nn.Sigmoid(),
            #nn.ReLU(True),
            #nn.Linear(32, 32),
            View((-1, 32, 1, 1)),
            #nn.ReLU(True),
            #nn.Sigmoid(),
            nn.ConvTranspose2d(32, 16, 4, stride=1),
            #nn.Sigmoid(),
            #nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, 4, 9, stride=1, padding=0),
            #nn.Sigmoid(),
            #nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(4, 1, 69, stride=1, padding=0),
            View((-1, size * size)),
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
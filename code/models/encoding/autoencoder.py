import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class TensorDataset(Dataset):
    """Custom Dataset for loading"""

    def __init__(self, X):
        self.X = torch.tensor(np.expand_dims(X, axis=1)).float()

    def __getitem__(self, item):
        return self.X[item, :, :, :]

    def __len__(self):
        return self.X.size(0)


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class CNNAutoencoder(nn.Module):

    def __init__(
            self,
            n_components=5
    ):
        super(CNNAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
                                                                    # (b, 1, 100, 100)
            nn.Conv2d(1, 4, 69, stride=1, padding=0),                # (b, 8, 13, 13)
            nn.Sigmoid(),                                           # (b, 8, 3, 3)
            nn.Conv2d(4, 16, 25, stride=1, padding=0),                # (b, 8, 13, 13)
            nn.Sigmoid(),                                           # (b, 8, 3, 3)
            nn.Conv2d(16, 128, 8, stride=1, padding=0),                # (b, 8, 13, 13)
            nn.Sigmoid(),                                           # (b, 8, 3, 3)
            View((-1, 128)),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, n_components)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_components, 32),
            nn.ReLU(True),
            nn.Linear(32, 128),
            View((-1, 128, 1, 1)),
            nn.ConvTranspose2d(128, 16, 8, stride=1),       # (b, 8, 8, 8)
            nn.Sigmoid(),                                          # (b, 8, 8, 8)
            nn.ConvTranspose2d(16, 4, 25, stride=1, padding=0),     # (b, 1, 100, 100)
            nn.Sigmoid(),                                          # (b, 8, 8, 8)
            nn.ConvTranspose2d(4, 1, 69, stride=1, padding=0),     # (b, 1, 100, 100)
            nn.Sigmoid(),                                          # (b, 8, 8, 8)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class NNAutoencoder(nn.Module):

    def __init__(
            self,
            input_size=100,
            n_components=5,
            layers=[1024, 128]
    ):
        super(NNAutoencoder, self).__init__()
        self.input_size = input_size
        self.n_components = n_components
        encoding_layers = []
        encoding_layers.append(View((-1, self.input_size ** 2)))
        prev = self.input_size ** 2
        for units in layers:
            encoding_layers.append(nn.Linear(prev, units))
            encoding_layers.append(nn.ReLU())
            prev = units
        encoding_layers.append(nn.Linear(prev, self.n_components))
        self.encoder = nn.Sequential(*encoding_layers)
        decoding_layers = []
        prev = self.n_components
        for units in layers[::-1]:
            decoding_layers.append(nn.Linear(prev, units))
            decoding_layers.append(nn.ReLU())
            prev = units
        decoding_layers.append(nn.Linear(prev, self.input_size ** 2))
        decoding_layers.append(View((-1, 1, self.input_size, self.input_size)))
        decoding_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoding_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Autoencoder:

    def __init__(
            self,
            model=CNNAutoencoder(),
            num_epochs=100,
            batch_size=8,
            learning_rate=0.01,
            seed=1,
            weight_decay=0.0001,
            loss=torch.nn.functional.binary_cross_entropy
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        torch.manual_seed(self.seed)
        self.model = model.cuda()
        self.loss = loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=weight_decay
        )

    def fit(self, X):
        self.dataset = TensorDataset(X)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.num_epochs):
            for x in self.dataloader:
                x = Variable(x).cuda()
                # forward step
                out = self.model(x)
                loss = self.loss(out, x)
                # backward step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.num_epochs, loss.data.item()))

    def transform(self, X):
        with torch.no_grad():
            X = torch.tensor(np.expand_dims(X, axis=1)).cuda().float()
            U = self.model.encoder(X).cpu().numpy()
        return U

    def inverse_transform(self, U):
        with torch.no_grad():
            U = torch.tensor(U).cuda().float()
            X = self.model.decoder(U).cpu().numpy()
            if len(X.shape) == 4:
                X = X[:, 0, :, :]
        return X
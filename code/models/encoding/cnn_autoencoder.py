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


class CNN(nn.Module):

    def __init__(
            self,
            n_components=5
    ):
        super(CNN, self).__init__()
        self.encoder = nn.Sequential(
                                                                    # (b, 1, 100, 100)
            nn.Conv2d(1, 32, 8, stride=8, padding=4),               # (b, 32, 13, 13)
            nn.ReLU(True),                                          # (b, 32, 13, 13)
            nn.MaxPool2d(4, stride=4),                              # (b, 32, 3, 3)
            nn.Conv2d(32, n_components, 3, stride=2, padding=1),    # (b, c, 2, 2)
            nn.ReLU(True),                                          # (b, c, 2, 2)
            nn.MaxPool2d(2, stride=1)                               # (b, c, 1, 1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_components, 32, 3, stride=1),      # (b, 32, 3, 3)
            nn.ReLU(True),                                          # (b, 32, 3, 3)
            nn.ConvTranspose2d(32, 8, 5, stride=2, padding=1),      # (b, 8, 7, 7)
            nn.ReLU(True),                                          # (b, 8, 7, 7)
            nn.ConvTranspose2d(8, 4, 3, stride=8, padding=1),       # (b, 4, 49, 49)
            nn.ReLU(True),                                          # (b, 4, 49, 49)
            nn.ConvTranspose2d(4, 1, 4, stride=2, padding=0),       # (b, 1, 100, 100)
            nn.Tanh()                                               # (b, 1, 100, 100)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class CNNAutoencoder:

    def __init__(
            self,
            model=CNN(),
            num_epochs=100,
            batch_size=8,
            learning_rate=0.01,
            seed=1,
            weight_decay=0.0001
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        torch.manual_seed(self.seed)
        self.model = model.cuda()
        self.criterion = nn.MSELoss()
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
                loss = self.criterion(out, x)
                # backward step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.num_epochs, loss.data.item()))

    def transform(self, X):
        with torch.no_grad():
            X = torch.tensor(np.expand_dims(X, axis=1)).cuda().float()
            U = self.model.encoder(X).cpu().numpy()[:, :, 0, 0]
        return U

    def inverse_transform(self, U):
        with torch.no_grad():
            U = torch.tensor(np.expand_dims(U, axis=[2, 3])).cuda().float()
            X = self.model.decoder(U).cpu().numpy()[:, 0, :, :]
        return X
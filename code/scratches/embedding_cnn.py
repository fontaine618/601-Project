import sys
import numpy as np
from models.encoding.autoencoder import Autoencoder, NNAutoencoder, CNNAutoencoder
from models.encoding.autoencoder import strikezone_dict_to_array, array_to_strikezone_dict
import matplotlib.pyplot as plt
from plot.utils import plot_pitches
import torch
import pickle

plt.style.use("seaborn")
sys.path.append('/home/simon/Documents/601-Project/code')

in_path = '/home/simon/Documents/601-Project/code/data/models/classifiers/'
in_files = [
    'umpire_balls_strikes_roc_auc_svc_klr.txt',
    'umpire_score_inning_roc_auc_svc_klr.txt',
    'umpire_pfx_x_z_auc_roc_svc_klr.txt',
    'umpire_pitchers_batters_auc_roc_svc_klr.txt',
]

szs = dict()
for in_file in in_files:
    szl = pickle.load(open(in_path + in_file, "rb"))
    grid_x, grid_y, sz = szl.compute_strike_zones()
    szs.update(sz)
# Transform all strike zones to arrays
# so that PCA can use them
groups, X = strikezone_dict_to_array(szs)

n_components = 15

nn = CNNAutoencoder(size=100, n_components=n_components).cuda()

loss = torch.nn.MSELoss()

opt = torch.optim.Adam(nn.parameters(), lr=0.0005)

torch.manual_seed(2)

encoder = Autoencoder(model=nn, optimizer=opt, loss=loss)

encoder.fit(X, num_epochs=30, batch_size=8)
for g in encoder.optimizer.param_groups:
    g['lr'] = 0.0001
encoder.fit(X, num_epochs=20, batch_size=8)

scores = encoder.score_samples(X)
stats = np.array([np.min(scores), np.quantile(scores, 0.05), np.mean(scores),
                  np.quantile(scores, 0.95), np.max(scores)])
print(stats.round(4))

plt.plot(encoder.losses_)
plt.show()



U = encoder.transform(X)

print(U)
print(np.cov(U.T))

Xr = encoder.inverse_transform(U)

szsr = array_to_strikezone_dict(groups, Xr)

print(encoder.score(X))
print(encoder.score_samples(X))




x_range = (grid_x.min(), grid_x.max())
z_range = (grid_y.max(), grid_y.min())

for k, lvls in enumerate(groups):
    if k > 9:
        break

    sz = szs[lvls]
    plot_pitches(x_range=x_range, z_range=z_range, sz=sz,
                 sz_type="contour", X=grid_x, Y=grid_y)

    sz = szsr[lvls].clip(0, 1)
    plot_pitches(x_range=x_range, z_range=z_range, sz=sz,
                 sz_type="uncertainty", X=grid_x, Y=grid_y)
    plt.title(str(lvls) + " reconstructed with NN(" + str(n_components) + ")")
    plt.show()




groups, X = strikezone_dict_to_array(szs)

X = torch.tensor(X).float()
print(X.size())
X = self.encoder(X)
print(X.size())
X = self.decoder(X)
print(X.size())

size = 100

print(X.size())
X = View((-1, 1, 100, 100))(X)
print(X.size())
X = nn.Conv2d(1, 4, 69, stride=1, padding=0)(X)
print(X.size())
X = nn.MaxPool2d(2, stride=2)(X)
print(X.size())
X = nn.ReLU(True)(X)
print(X.size())
X = nn.Conv2d(4, 16, 9, stride=1, padding=0)(X)
print(X.size())
X = nn.MaxPool2d(2, stride=2)(X)
print(X.size())
X = nn.ReLU(True)(X)
print(X.size())
X = nn.Conv2d(16, 32, 4, stride=1, padding=0)(X)
print(X.size())
X = View((-1, 32))(X)
print(X.size())
X = nn.Linear(32, n_components)(X)
print(X.size())
# out
X = nn.Linear(n_components, 32)(X)
print(X.size())
X = View((-1, 32, 1, 1))(X)
print(X.size())
X = nn.ConvTranspose2d(32, 16, 4, stride=1, padding=0)(X)
print(X.size())
X = nn.Upsample(scale_factor=2)(X)
print(X.size())
X = nn.ConvTranspose2d(16, 4, 9, stride=1, padding=0)(X)
print(X.size())
X = nn.Upsample(scale_factor=2)(X)
print(X.size())
X = nn.ConvTranspose2d(4, 1, 69, stride=1, padding=0)(X)
print(X.size())
X = View((-1, size * size))(X)
print(X.size())



self = encoder.model


with torch.no_grad():
    out = encoder.model(X)
    loss = encoder.loss(out, X)
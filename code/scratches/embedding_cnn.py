import sys
import numpy as np
from models.encoding.autoencoder import Autoencoder, NNAutoencoder, strikezone_dict_to_array, array_to_strikezone_dict
from models.encoding.autoencoder_old import CNNAutoencoder
import matplotlib.pyplot as plt
from plot.utils import plot_pitches
import torch
import pickle

plt.style.use("seaborn")
sys.path.extend(['/home/simon/Documents/601-Project/code'])
with open("./data/models/classifiers/umpire_balls_strikes_roc_auc_svc_klr.txt", "rb") as f:
    szl = pickle.load(f)

grid_x, grid_y, szs = szl.compute_strike_zones()
x_range = (grid_x.min(), grid_x.max())
z_range = (grid_y.max(), grid_y.min())

groups, X = strikezone_dict_to_array(szs)

n_components = 200

nn = NNAutoencoder(layers=[1028, 128], n_components=n_components).cuda()

loss = torch.nn.MSELoss()

opt = torch.optim.SGD(nn.parameters(), lr=100)

torch.manual_seed(2)

encoder = Autoencoder(model=nn, optimizer=opt, loss=loss)

encoder.fit(X, num_epochs=100)

U = encoder.transform(X)

print(U)
print(np.cov(U.T) * 1e6)

Xr = encoder.inverse_transform(U)

szsr = array_to_strikezone_dict(groups, Xr)

print(encoder.score(X))
print(encoder.score_samples(X))


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
# %%
# Imports
import numpy as np
import pandas as pd
import itertools as it
from joblib import Parallel, delayed
import pickle
import sys

from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics import mean_squared_error as mse

# -------------- Setup -------------------

sys.path.append('/path/to/code')

in_path = '/path/to/load/'
in_file = 'umpire_balls_strikes_brier_svc_klr.txt'

out_path = '/path/to/dump/'
out_file = 'appropriateFile'

max_components = 10
n_components = range(1, max_components+1)
alphas = [0.1, 1, 2, 10, 100]
gammas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
kernel_pca_arguments = list(it.product(alphas, gammas))

n_jobs = -1

# Read the strike_zones from the pickle file
# This expects the strike zones to be a dictionary
szl = pickle.load(open(in_path + in_file, "rb"))
groups = szl.groups
grid_x, grid_y, strike_zones = szl.compute_strike_zones()

# -------------- User Defined Functions -------------------

def fit_kernel_pca(sz_array, alpha, gamma, nc):

    kernel_pca = KernelPCA(alpha=alpha, gamma=gamma,
                           n_components=n, kernel='rbf',
                           fit_inverse_transform=True)
    transformed = kernel_pca.fit_transform(sz_array)
    inverse_transformed = kernel_pca.inverse_transform(transformed)
    losses = []
    for i in range(transformed.shape[0]):
        losses.append(mse(sz_array[i, :], inverse_transformed[i, :]))

    return np.mean(losses), alpha, gamma, inverse_transformed

# -------------- Procedure -------------------

# The result data frames
results_pca = pd.DataFrame(
    data=np.inf * np.ones(
        shape=(len(groups), len(n_components))
    ),
    columns=n_components,
    index=groups
)

results_kernel_pca = pd.DataFrame(
    data=np.inf * np.ones(
        shape=(len(groups), len(n_components))
    ),
    columns=n_components,
    index=groups
)
# Transform all strike zones to arrays
# so that PCA can use them
sz_array = np.ones(shape=(len(groups), 10000))
counter = 0
for group, sz in strike_zones.items():
    sz_array[counter, :] = sz.ravel()
    counter += 1

for n in n_components:
    out = Parallel(n_jobs=2)(
        delayed(fit_kernel_pca)(sz_array,
                                alpha,
                                gamma,
                                n
                                ) for alpha, gamma in kernel_pca_arguments
    )
    bl = np.inf
    for t in out:
        if t[0] < bl:
            bl = t[0]
            best_alpha = t[1]
            best_gamma = t[2]
            kernel_pca_inverse_transformed = t[3]

    counter = 0
    for group in groups:
        loss = mse(
            sz_array[counter, :], kernel_pca_inverse_transformed[counter, :]
        )
        results_kernel_pca.at[group, n] = loss
        counter += 1

    pca = PCA(n_components=n)
    transformed = pca.fit_transform(sz_array)
    inverse_transformed = pca.inverse_transform(transformed)
    counter = 0
    for group in groups:
        loss = mse(
            sz_array[counter, :], inverse_transformed[counter, :]
        )
        results_pca.at[group, n] = loss
        counter += 1

# Potentially pickle at some point
# pickle.dump(results, open(out_path + out_file, "wb"))

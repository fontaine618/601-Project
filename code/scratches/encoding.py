# %%
# Imports
import numpy as np
import pandas as pd
import itertools as it
from joblib import Parallel, delayed
import pickle
import sys
import matplotlib.pyplot as plt
from plot.utils import plot_pitches

from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics import mean_squared_error as mse

from models.encoding.autoencoder import strikezone_dict_to_array, array_to_strikezone_dict

# -------------- Setup -------------------

sys.path.append('/home/simon/Documents/601-Project/code')

in_path = './data/models/classifiers/'
in_file = 'umpire_balls_strikes_roc_auc_svc_klr.txt'

out_path = './data/models/encoding/'
out_file = 'umpire_balls_strikes_k-pca_results.txt'
out_file_fit = 'umpire_balls_strikes_fit.txt'

max_components = 15
n_components = range(1, max_components+1)
alphas = np.logspace(-8, -1, 10)
gammas = np.logspace(-4, -1, 10)
kernel_pca_arguments = list(it.product(alphas, gammas))

n_jobs = 10

# Read the strike_zones from the pickle file
# This expects the strike zones to be a dictionary
szl = pickle.load(open(in_path + in_file, "rb"))
grid_x, grid_y, strike_zones = szl.compute_strike_zones()
groups, sz_array = strikezone_dict_to_array(strike_zones)
# Transform all strike zones to arrays
# so that PCA can use them
groups, sz_array = strikezone_dict_to_array(strike_zones)

# -------------- User Defined Functions -------------------

def fit_kernel_pca(sz_array, alpha, gamma, nc):

    kernel_pca = KernelPCA(alpha=alpha, gamma=gamma,
                           n_components=nc, kernel='rbf',
                           fit_inverse_transform=True)
    transformed = kernel_pca.fit_transform(sz_array)
    inverse_transformed = kernel_pca.inverse_transform(transformed)
    losses = mse(sz_array.T, inverse_transformed.T, multioutput='raw_values').T

    return losses, alpha, gamma, transformed, inverse_transformed, kernel_pca

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

for n in n_components:
    out = Parallel(n_jobs=n_jobs)(
        delayed(fit_kernel_pca)(sz_array,
                                alpha,
                                gamma,
                                n
                                ) for alpha, gamma in kernel_pca_arguments
    )
    bl = np.inf
    for l, a, g, t, s, _ in out:
        if l.mean() < bl:
            bl = l.mean()
            best_alpha = a
            best_gamma = g
            kernel_pca_inverse_transformed = g
            losses = l
    print(n, best_alpha, best_gamma)

    results_kernel_pca.at[:, n] = losses

    pca = PCA(n_components=n)
    transformed = pca.fit_transform(sz_array)
    inverse_transformed = pca.inverse_transform(transformed)
    losses = mse(sz_array.T, inverse_transformed.T, multioutput='raw_values').T
    results_pca.at[:, n] = losses

# Potentially pickle at some point
with open(out_path + out_file, "wb") as f:
    pickle.dump((results_pca, results_kernel_pca), f)


# --------------- Plot MSEs ------------------------
stats_pca = results_pca.agg(["min", "mean", "max", "std"], axis=0).T
stats_kernel_pca = results_kernel_pca.agg(["min", "mean", "max", "std"], axis=0).T

plt.style.use("seaborn")
fig = plt.figure()
plt.semilogy(True)
plt.plot(n_components, stats_pca["mean"], label="PCA")
plt.fill_between(n_components, stats_pca["min"], stats_pca["max"], alpha=0.2)
plt.plot(n_components, stats_kernel_pca["mean"], label="KernelPCA")
plt.fill_between(n_components, stats_kernel_pca["min"], stats_kernel_pca["max"], alpha=0.2)
plt.legend(title="Model")
plt.xlabel("Number of components")
plt.ylabel("Reconstruction MSE")
plt.tight_layout()
plt.show()

# --------------- Plot reconstructed Szs ---------------

losses, alpha, gamma, U, Xr, fit = fit_kernel_pca(sz_array, 1e-8, 0.001, 10)

szsr = array_to_strikezone_dict(groups, Xr)
x_range = (grid_x.min(), grid_x.max())
z_range = (grid_y.max(), grid_y.min())

for k, (gr, sz, szr) in enumerate(zip(groups, strike_zones.values(), szsr.values())):
    if k > 9:
        break
    plot_pitches(x_range=x_range, z_range=z_range, sz=sz,
                 sz_type="uncertainty", X=grid_x, Y=grid_y)
    plot_pitches(x_range=x_range, z_range=z_range, sz=szr,
                 sz_type="contour", X=grid_x, Y=grid_y, levels=[0.25, 0.5, 0.75])
    plt.title(str(gr) + " reconstructed with KPCA(10)")
    plt.show()

# Potentially pickle at some point
with open(out_path + out_file_fit, "wb") as f:
    pickle.dump((fit, U, list(strike_zones.keys()), x_range, z_range), f)
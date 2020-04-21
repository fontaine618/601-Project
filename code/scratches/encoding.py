# %%
# Imports
import numpy as np
import pandas as pd
import itertools as it
from joblib import Parallel, delayed
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib
from plot.utils import *

from sklearn.decomposition import KernelPCA, PCA
from sklearn.metrics import mean_squared_error as mse

from models.encoding.autoencoder import strikezone_dict_to_array, array_to_strikezone_dict
# -------------- Setup -------------------

sys.path.append('/home/simon/Documents/601-Project/code')

in_path = '/home/simon/Documents/601-Project/code/data/models/classifiers/'
in_files = [
	'umpire_balls_strikes_roc_auc_svc_klr.txt',
	'umpire_score_inning_roc_auc_svc_klr.txt',
	'umpire_pfx_x_z_auc_roc_svc_klr.txt',
	'umpire_pitchers_batters_auc_roc_svc_klr.txt',
]

out_path = '/home/simon/Documents/601-Project/code/data/models/encoding/'
out_results = 'all_results.txt'
out_fit = 'all_fit.txt'

max_components = 15
n_components = range(1, max_components + 1)
alphas = np.logspace(-8, -8, 1)
gammas = np.logspace(-4, -1, 10)
kernel_pca_arguments = list(it.product(alphas, gammas))

n_jobs = 10

# Read the strike_zones from the pickle file
# This expects the strike zones to be a dictionary
strike_zones = dict()
for in_file in in_files:
	szl = pickle.load(open(in_path + in_file, "rb"))
	grid_x, grid_y, sz = szl.compute_strike_zones()
	strike_zones.update(sz)
# Transform all strike zones to arrays
# so that PCA can use them
groups, sz_array = strikezone_dict_to_array(strike_zones)


# -------------- User Defined Functions -------------------

def fit_kernel_pca(sz_array, alpha, gamma, nc, return_fit=True):
	kernel_pca = KernelPCA(alpha=alpha, gamma=gamma,
						   n_components=nc, kernel='rbf',
						   fit_inverse_transform=True)
	transformed = kernel_pca.fit_transform(sz_array)
	inverse_transformed = kernel_pca.inverse_transform(transformed)
	losses = mse(sz_array.T, inverse_transformed.T, multioutput='raw_values').T
	if return_fit:
		return losses, alpha, gamma, transformed, inverse_transformed, kernel_pca
	else:
		return losses, alpha, gamma, transformed, inverse_transformed


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
	out = Parallel(n_jobs=n_jobs, prefer="threads")(
		delayed(fit_kernel_pca)(sz_array,
								alpha,
								gamma,
								n,
								False
								) for alpha, gamma in kernel_pca_arguments
	)
	bl = np.inf
	for l, a, g, t, s in out:
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
with open(out_path + out_results, "wb") as f:
	pickle.dump((results_pca, results_kernel_pca), f)

# --------------- Plot MSEs ------------------------

with open(out_path + out_results, "rb") as f:
	results_pca, results_kernel_pca = pickle.load(f)


def percentile(n):
	def percentile_(x):
		return x.quantile(n)

	percentile_.__name__ = 'percentile_{:.0f}'.format(n * 100)
	return percentile_


stats_pca = results_pca.agg([
	np.min, np.mean, np.max, np.std, percentile(0.05), percentile(0.95)
], axis=0).T
stats_kernel_pca = results_kernel_pca.agg([
	np.min, np.mean, np.max, np.std, percentile(0.05), percentile(0.95)
], axis=0).T



plt.style.use("seaborn")
cols = ["#00274c", "#ffcb05"]
fig = plt.figure(figsize=(6, 4))
plt.plot(n_components, stats_pca["mean"], label="PCA", color=cols[0])
# plt.fill_between(n_components, stats_pca["min"], stats_pca["max"], alpha=0.2)
plt.fill_between(n_components, stats_pca["percentile_5"],
				 stats_pca["percentile_95"], alpha=0.2, color=cols[0])
plt.plot(n_components, stats_kernel_pca["mean"], label="KernelPCA", color=cols[1])
# plt.fill_between(n_components, stats_kernel_pca["min"], stats_kernel_pca["max"], alpha=0.2)
plt.fill_between(n_components, stats_kernel_pca["percentile_5"],
				 stats_kernel_pca["percentile_95"], alpha=0.2, color=cols[1])
leg = plt.legend(title="Average MSE", frameon=True, loc="upper right")
leg._legend_box.align = "left"
plt.title("Encoders' Prediction Error by Number of Components",
		  loc="left", fontweight="bold")
plt.xlabel("Number of components")
plt.ylabel("Prediction MSE")
plt.tight_layout()

plt.show()

plt.savefig("./fig/encoder_results.pdf")


# --------------- Plot reconstructed Szs ---------------

losses, alpha, gamma, U, Xr, fit = fit_kernel_pca(sz_array, 1e-8, 0.001, 10)

szsr = array_to_strikezone_dict(groups, Xr)
x_range = (grid_x.min(), grid_x.max())
z_range = (grid_y.max(), grid_y.min())

for k, (gr, sz, szr) in enumerate(zip(groups, strike_zones.values(), szsr.values())):
	if k < 379 or k > 390:
		continue
	plot_pitches(x_range=x_range, z_range=z_range, sz=sz,
				 sz_type="uncertainty", X=grid_x, Y=grid_y)
	plot_pitches(x_range=x_range, z_range=z_range, sz=szr,
				 sz_type="contour", X=grid_x, Y=grid_y, levels=[0.25, 0.5, 0.75])
	plt.title(str(gr) + " reconstructed with KPCA(10)")
	plt.show()

# Potentially pickle at some point
with open(out_path + out_fit, "wb") as f:
	pickle.dump((fit, U, list(strike_zones.keys()), x_range, z_range), f)



# ------------------ Encoding plot ------------------

gr = ("Joe West", "b_count_[0,2]", "s_count_(1,2]")
sz = strike_zones[gr]
szr = szsr[gr]
plt.style.use("seaborn")

components_names = [
    "Smaller",
    "Uncertain",
    "High inside excluded",
    "Wide bottom",
    "Wide middle",
    "Wide top",
    "NW/SE diagonal",
    "Irregular 1",
    "Irregular 2",
    "Irregular 3",
]


N = 128
vals = np.ones((N*2, 4))
vals[:N, 0] = np.linspace(0/256, 1, N)
vals[:N, 1] = np.linspace(39/256, 1, N)
vals[:N, 2] = np.linspace(76/256, 1, N)
vals[N:, 0] = np.linspace(1, 255/256, N)
vals[N:, 1] = np.linspace(1, 203/256, N)
vals[N:, 2] = np.linspace(1, 2/256, N)
newcmp = matplotlib.colors.ListedColormap(vals)
newcmp.set_bad(color='white')


fig = plt.figure(figsize=(10, 3.7))
# original
plt.subplot(131)
plt.imshow(sz, extent=(*x_range, *z_range[::-1]), alpha=sz.astype(float))
plt.plot(*batter_outline(x=-0.8), scalex=False, scaley=False, color="white", linewidth=4)
plt.plot(*strike_zone(), scalex=False, scaley=False, color="white", linewidth=1, linestyle="--")
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.gca().set_xlim(*x_range)
plt.gca().set_ylim(*z_range[::-1])
plt.title("Original strike zone")


plt.subplot(132)
plt.imshow(U[groups.index(gr), :].reshape((-1, 1)), cmap=newcmp, vmin=-0.4, vmax=0.4)
plt.grid(None)
plt.colorbar()
plt.yticks(range(10), components_names)
plt.xticks([])
plt.title("Encoding")

plt.subplot(133)
plt.imshow(sz, extent=(*x_range, *z_range[::-1]), alpha=sz.astype(float))
plt.contour(grid_x, grid_y, szr, extent=(*x_range, *z_range[::-1]), levels=[0.5])
plt.plot(*batter_outline(x=-0.8), scalex=False, scaley=False, color="white", linewidth=4)
plt.plot(*strike_zone(), scalex=False, scaley=False, color="white", linewidth=1, linestyle="--")
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.gca().set_xlim(*x_range)
plt.gca().set_ylim(*z_range[::-1])
plt.title("Reconstructed strike zone")

plt.tight_layout()

plt.savefig("./fig/encoding.pdf")

plt.show()


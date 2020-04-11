import sys
from data.pitchfx import PitchFxDataset
from models.classification.StrikeZoneLearner import StrikeZoneLearner
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from plot.utils import plot_pitches
plt.style.use("seaborn")

sys.path.extend(['/home/simon/Documents/601-Project/code'])
with open("./data/models/classifiers/umpire_balls_strikes_accuracy.txt", "rb") as f:
    szl = pickle.load(f)

pitchfx = PitchFxDataset()
df = pitchfx.group_by(
    umpire_HP="all",
    b_count=[0, 2, 3],
    s_count=[0, 1, 2]
)

scores = []
fits = []
n_obs = []
groups = []

for group, pitches in df:
    groups.append(group)
    scores.append(szl.scores[group])
    fits.append(type(szl.fits[group]).__name__)
    n_obs.append(len(pitches))

results = pd.DataFrame({
    "score": scores, "model": fits, "n_obs": n_obs
})
results.index = groups

plt.style.use("seaborn")
plt.figure()
models = results["model"].unique()
for k, model in enumerate(models):
    plt.scatter(
        x=results["n_obs"][results["model"] == model],
        y=results["score"][results["model"] == model],
        label=model
    )
plt.legend()
plt.show()

# fix old version
szl_ = StrikeZoneLearner(scoring="accuracy")
szl_.groups = set([gr for gr, _ in df])
szl_.fits = szl.fits

grid_x, grid_y, szs = szl_.compute_strike_zones()
x_range = (grid_x.min(), grid_x.max())
z_range = (grid_y.max(), grid_y.min())

levels = ("Angel Hernandez", "b_count_[0,2]", "s_count_(1,2]")
sz = szs[levels]

for levels, sz in szs.items():
    pitches = df.get_group(levels)
    plot_pitches(
        pitches=pitches,
        x_range=x_range, z_range=z_range,
        sz=sz, sz_type="heatmap",
    )
    plt.title(results["model"][levels] + str(levels))
    plt.show()


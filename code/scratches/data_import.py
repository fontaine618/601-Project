import sys
sys.path.extend(['/home/simon/Documents/601-Project/code'])
from data import PitchFxDataset
import matplotlib.pyplot as plt
import pandas as pd
from plot.batter_outline import batter_outline

pitchfx = PitchFxDataset()
pd.crosstab(pitchfx.pitchfx["type"], pitchfx.pitchfx["type_from_sz"])
df = pitchfx.group_by(
    umpire_HP="all",
    stand="all",
)

# to iterate through all:
for levels, d in df:
    print(len(d), levels)

plt.scatter(pitchfx.pitchfx["pz"][:1000], pitchfx.pitchfx["pz_std"][:1000])
plt.hist(pitchfx.pitchfx["pz_std"] - pitchfx.pitchfx["pz"])
plt.show()

from models.classification.kernel_logistic_regression import KernelLogisticRegression
import numpy as np
plt.style.use("seaborn")
it = iter(df)
x, y = np.meshgrid(
    np.linspace(-1.2, 1.2, num=100),
    np.linspace(4, 1, num=100)
)
X = np.concatenate([x.reshape((-1, 1)), y.reshape((-1, 1))], axis=1)
klr = KernelLogisticRegression(
    gamma=1.
)

levels, pitches = next(it)
klr.fit(
    X=pitches[["px_std", "pz_std"]].to_numpy(),
    y=pitches["type"].to_numpy()
)
pred = klr.predict_proba(X)
pred_grid = pred.reshape((100, 100))
plt.imshow(pred_grid)
plt.title(levels)
plt.show()

plt.scatter(pitches["px_std"], pitches["pz_std"])
plt.plot(*batter_outline(x = -1))
plt.show()
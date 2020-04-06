import sys
sys.path.extend(['/home/simon/Documents/601-Project/code'])
from data.pitchfx import PitchFxDataset
from models.classification.strikezone_learner import StrikezoneLearner
from models.classification.kernel_logistic_regression import KernelLogisticRegression
import matplotlib.pyplot as plt
from plot.utils import batter_outline, strike_zone, labeled_pitches
from pygam import LogisticGAM, te
from sklearn.svm import SVC
plt.style.use("seaborn")


pitchfx = PitchFxDataset()
df = pitchfx.group_by(
    umpire_HP="all",
    b_count=[0, 2, 3],
    s_count=[0, 1, 2]
)

# classifier = KernelLogisticRegression(
#     gamma=1.
# )

# classifier = LogisticGAM(
#     te(0, 1, n_splines=12, lam=0.1)
# )

C = 0.7
gamma = 1.5

classifier = SVC(
    C=C,
    gamma=gamma,
    probability=True,
    class_weight="balanced"
)



self = StrikezoneLearner(df, classifier, x_range=(-2, 2), y_range=(5, 0.5))

# self.fit_all()

self.predict_strikezone_all()

levels = ("Angel Hernandez", "b_count_[0,2]", "s_count_(1,2]")
sz = self.strikezone[levels]
for levels, sz in self.strikezone.items():
    plt.imshow(sz, extent=(*self.x_range, *self.y_range[::-1]))
    plt.title(levels)
    plt.plot(*batter_outline(), scalex=False, scaley=False)
    plt.plot(*strike_zone(), scalex=False, scaley=False)
    pitches = df.get_group(levels)
    xb, zb, xs, zs = labeled_pitches(pitches)
    plt.scatter(xb, zb, label="Ball")
    plt.scatter(xs, zs, label="Strike")
    plt.legend()
    plt.show()

self.cv_all(scoring="accuracy")

plt.scatter(
    x=self.counts.values(),
    y=self.cv_accuracy.values()
)
plt.xlabel("Nb. pitches")
plt.ylabel("CV(5) accuracy")
plt.title("SVM (C={}, gamma={})".format(C, gamma))
plt.show()

# n_folds = 5
# n_jobs=-1
# prefer="threads"
levels = ("Adam Hamari", "L")
pitches = self.pitches.get_group(levels)

import numpy as np
X = pitches[["px_std", "pz_std"]].to_numpy()
y = pitches["type_01"].to_numpy().reshape((-1))
fit = classifier.fit(X, y)
np.mean(classifier.predict(X) == y)
cv = classifier.gridsearch(X, y)
cv.summary()
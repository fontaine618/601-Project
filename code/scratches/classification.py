import sys
sys.path.extend(['/home/simon/Documents/601-Project/code'])
from data.pitchfx import PitchFxDataset
from models.classification.strikezone_learner import StrikezoneLearner
from models.classification.kernel_logistic_regression import KernelLogisticRegression
import matplotlib.pyplot as plt
from plot.utils import batter_outline, strike_zone


pitchfx = PitchFxDataset()
df = pitchfx.group_by(
    umpire_HP="all",
    stand="all",
)

klr = KernelLogisticRegression(
    gamma=1.
)

self = StrikezoneLearner(df, klr, x_range=(-2, 2), y_range=(5, 0.5))

self.fit_all()

self.predict_strikezone_all()

plt.style.use("seaborn")

for levels, sz in self.strikezone.items():
    plt.imshow(sz, extent=(*self.x_range, *self.y_range[::-1]))
    plt.title(levels)
    plt.plot(*batter_outline(), scalex=False, scaley=False)
    plt.plot(*strike_zone(), scalex=False, scaley=False)
    plt.show()
    break

self.cv_all()

plt.style.use("seaborn")
plt.scatter(
    x=self.counts.values(),
    y=self.cv_accuracy.values()
)
plt.xlabel("Nb. pitches")
plt.ylabel("CV(5) accuracy")
plt.title("Gaussian Kernel Logistic Regression")
plt.show()

n_folds = 5
n_jobs=-1
prefer="threads"
levels = ("Adam Hamari", "L")
pitches = self.pitches.get_group(levels)


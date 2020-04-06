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

self = StrikezoneLearner(df, klr, x_range=(-2.5, 2.5), y_range=(6, 0))

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
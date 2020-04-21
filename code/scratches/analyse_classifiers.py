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
with open("./data/models/classifiers/umpire_balls_strikes_roc_auc_svc_klr.txt", "rb") as f:
	szl = pickle.load(f)

pitchfx = PitchFxDataset()
df = pitchfx.group_by(
	umpire_HP="all",
	b_count=[0, 2, 3],
	s_count=[0, 1, 2]
)

pitchfx.pitchfx["pfx_x"] = np.where(
	pitchfx.pitchfx["stand"] == "L",
	-pitchfx.pitchfx["pfx_x"],
	pitchfx.pitchfx["pfx_x"]
)
df = pitchfx.group_by(
	umpire_HP="all",
	pfx_x=[-60, 0, 60],
	pfx_z=[-20, 5, 20]
)

df = pitchfx.group_by(
	umpire_HP="all",
	p_throws="all",
	stand="all"
)

pitchfx.pitchfx["score_diff_b_p"] = pitchfx.pitchfx["b_score"] - pitchfx.pitchfx["p_score"]

df = pitchfx.group_by(
	umpire_HP="all",
	score_diff_b_p=[-25, -2, 1, 25],
	inning=[1, 6, 18]
)

scores = []
fits = []
n_obs = []
groups = []
params = []

for group, pitches in df:
	groups.append(group)
	scores.append(szl.scores[group])
	fits.append(type(szl.fits[group]).__name__)
	n_obs.append(len(pitches))
	params.append(szl.params[group])

results = pd.DataFrame({
	"score": scores, "model": fits, "n_obs": n_obs, "params": params
})
results.index = groups


plt.figure(figsize=(6, 6))
cols = ["#00274c", "#ffcb05"]
models = results["model"].unique()
for k, model in enumerate(models):
	plt.scatter(
		x=results["n_obs"][results["model"] == model],
		y=results["score"][results["model"] == model],
		label=model,
		color=cols[k],
		marker="oo"[k]
	)
leg = plt.legend(loc="lower right", title="Selected classifier", frameon=True)
leg._legend_box.align = "left"
plt.xlabel("Number of pitches")
plt.ylabel("AUROC")
plt.title("CV(5) Score of the Selected Classifier for each Pitch Subset",
		  loc="left",fontweight="bold")
plt.tight_layout()
plt.savefig("./fig/classifiers_cv_results.pdf")
plt.show()

# # fix old version
# szl_ = StrikeZoneLearner(scoring="accuracy")
# szl_.groups = set([gr for gr, _ in df])
# szl_.fits = szl.fits

grid_x, grid_y, szs = szl.compute_strike_zones()
x_range = (grid_x.min(), grid_x.max())
z_range = (grid_y.max(), grid_y.min())

levels = ("Angel Hernandez", "b_count_[0,2]", "s_count_(1,2]")
sz = szs[levels]

for k, (group, sz) in enumerate(szs.items()):
	if k > 10:
		break
	pitches = df.get_group(group)
	plot_pitches(
		pitches=pitches,
		x_range=x_range, z_range=z_range,
		sz=sz, sz_type="uncertainty",
	)
	plt.title(results["model"][group] + str(results["params"][group]) + "\n" + str(group))
	plt.show()

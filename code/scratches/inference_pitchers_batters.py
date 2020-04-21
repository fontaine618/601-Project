import sys
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import matplotlib
from data.pitchfx import PitchFxDataset

# ---------------- COLORS ---------------------

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


N = 20
vals = np.ones((N * N, 4))
vals[:N, 0] = np.linspace(255/256, 0/256, N)
vals[:N, 1] = np.linspace(203/256, 39/256, N)
vals[:N, 2] = np.linspace(2/256, 76/256, N)
vals[N:, :] = vals[N-1, :]
newcmp16 = matplotlib.colors.ListedColormap(vals)
newcmp16.set_bad(color='white')

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

# ---------------- SETUP ------------------

plt.style.use("seaborn")
sys.path.extend(['/home/simon/Documents/601-Project/code'])
encoder_path = "./data/models/encoding/all_fit.txt"


pitchfx = PitchFxDataset()

# Balls and strike counts

df = pitchfx.group_by(
	umpire_HP="all",
	p_throws="all",
    stand="all"
)


with open(encoder_path, "rb") as f:
    _, embeddings, groups, _, _ = pickle.load(f)

ids = [groups.index(gr) for gr, _ in df if gr in groups]

embeddings = embeddings[ids, :]
groups = [groups[i] for i in ids]

df = pd.DataFrame(embeddings, index=pd.MultiIndex.from_tuples(groups)).reset_index()
df.columns = ["umpire", "pitcher", "batter", *["c"+str(i) for i in range(10)]]


# ------------ MANOVA -------------------

manova = MANOVA.from_formula("c0+c1+c2+c3+c4+c5+c6+c7+c8+c9~umpire+pitcher*batter", data=df)
table = manova.mv_test()

res = pd.DataFrame({
    term: table.results[term]["stat"].iloc[0]
    for term in table.results
}).T

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

# --------------- Univariate ANOVAs ----------------

p_values = pd.DataFrame(
    {
        components_names[i]:
        sm.stats.anova_lm(ols("c" + str(i) + "~umpire+pitcher*batter", data=df).fit(), typ=2)["PR(>F)"][1:4]
        for i in range(10)
    }
)

coefs = pd.DataFrame(
    {
        components_names[i]:
        ols("c" + str(i) + "~umpire+pitcher*batter", data=df).fit().params[-3:]
        for i in range(10)
    }
)

umpires = pd.DataFrame(
    {
        components_names[i]:
        ols("c" + str(i) + "~umpire+pitcher*batter", data=df).fit().params[:-3]
        for i in range(10)
    }
)


fig = plt.figure(figsize=(8, 4))
plt.imshow(p_values, cmap=newcmp16, vmin=0, vmax=1)
cbar = plt.colorbar()
plt.grid(False)
plt.xticks(range(10), p_values.columns, rotation=90)
plt.yticks(range(3), ["Pitcher", "Batter", "Pitcher-batter interaction"])
plt.title("Univariate ANOVAs: p-values", loc="left", fontweight="bold")
fig.tight_layout()
# plt.show()
plt.savefig("./fig/ols_pitcher_batter_pvalues.pdf")

threshold = 0.05 / 3 # Bonferroni
fig = plt.figure(figsize=(8, 3.8))
coefs[p_values.to_numpy()[:, :] > threshold] = np.nan
plt.imshow(coefs, cmap=newcmp, vmin=-0.2, vmax=0.2)
cbar = plt.colorbar()
plt.grid(False)
plt.xticks(range(10), p_values.columns, rotation=90)
plt.yticks(range(3), ["Pitcher R", "Batter R", "Pitcher R:Batter R"])
plt.title("Univariate ANOVAs significant effects:\nestimated differences from L pitcher and L batter",
          loc="left", fontweight="bold")
fig.tight_layout()
#plt.show()
plt.savefig("./fig/ols_pitcher_batter_diffs.pdf")







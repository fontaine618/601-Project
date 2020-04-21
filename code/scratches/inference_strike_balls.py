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
    b_count=[0, 2, 3],
    s_count=[0, 1, 2]
)


with open(encoder_path, "rb") as f:
    _, embeddings, groups, _, _ = pickle.load(f)

ids = [groups.index(gr) for gr, _ in df if gr in groups]

embeddings = embeddings[ids, :]
groups = [groups[i] for i in ids]

df = pd.DataFrame(embeddings, index=pd.MultiIndex.from_tuples(groups)).reset_index()
df.columns = ["umpire", "ball_count", "strike_count", *["c"+str(i) for i in range(10)]]
df["ball_count"] = df["ball_count"].str.replace("b_count_", "")
df["strike_count"] = df["strike_count"].str.replace("s_count_", "")


# ------------ Check 2nd component ---------------

n = [len(df.get_group(gr)) for gr in groups]

c = 1

y = embeddings[:, c-1]

plt.scatter(n, y)
plt.show()

# ------------ MANOVA -------------------

manova = MANOVA.from_formula("c0+c1+c2+c3+c4+c5+c6+c7+c8+c9~umpire+ball_count*strike_count", data=df)

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
        sm.stats.anova_lm(ols("c" + str(i) + "~ball_count*strike_count", data=df).fit(), typ=2)["PR(>F)"][:3]
        for i in range(10)
    }
)

coefs = pd.DataFrame(
    {
        components_names[i]:
        ols("c" + str(i) + "~ball_count*strike_count", data=df).fit().params[-3:]
        for i in range(10)
    }
)

umpires = pd.DataFrame(
    {
        components_names[i]:
        ols("c" + str(i) + "~ball_count*strike_count", data=df).fit().params[:-3]
        for i in range(10)
    }
)


fig = plt.figure(figsize=(8, 4))
plt.imshow(p_values, cmap=newcmp16, vmin=0, vmax=1)
cbar = plt.colorbar()
plt.grid(False)
plt.xticks(range(10), p_values.columns, rotation=90)
plt.yticks(range(3), ["Ball count", "Strike count", "Ball-strike interaction"])
plt.title("Univariate ANOVAs: p-values", loc="left", fontweight="bold")
fig.tight_layout()
#plt.show()
plt.savefig("./fig/ols_pvalues.pdf")

threshold = 0.05 / 3 # Bonferroni
fig = plt.figure(figsize=(8, 3.8))
coefs[p_values.to_numpy()[:, :] > threshold] = np.nan
plt.imshow(coefs, cmap=newcmp, vmin=-0.3, vmax=0.3)
cbar = plt.colorbar()
plt.grid(False)
plt.xticks(range(10), p_values.columns, rotation=90)
plt.yticks(range(3), ["Ball [0,2]", "Strike [0,1]", "Ball [0,2]:Strike [0,1]"])
plt.title("Univariate ANOVAs significant effects:\nestimated differences from 3-2 count",
          loc="left", fontweight="bold")
fig.tight_layout()
#plt.show()
plt.savefig("./fig/ols_diffs.pdf")

















# --------------------- LMM -------------------
df_long = df.melt(
    id_vars=["umpire", "ball_count", "strike_count"],
    value_vars=["c"+str(i) for i in range(10)],
    var_name="component",
    value_name="value"
)
df_long["umpire:component"] = [a + ":" + b for a, b in zip(df_long["umpire"], df_long["component"])]

X = df_long[["umpire", "ball_count", "strike_count", "component", "umpire:component"]].to_numpy()
y = df_long["value"].to_numpy()

lmm = smf.mixedlm(
    "value ~ 1 + ball_count * strike_count * component",
    data=df_long,
    groups="umpire:component",
)
lmm = lmm.fit()
print(lmm.summary())

ord = [0, 1, 2, 12]
for i in range(0, 9):
    ord.extend([3 + i, 13 + i, 22 + i, 31 + i])

params = lmm.params[ord]
pvalues = lmm.pvalues[ord]

results = pd.DataFrame(
    {"params": params.to_numpy(), "pvalues": pvalues.to_numpy()},
    index=pd.MultiIndex.from_product([
        ["c"+str(i) for i in range(10)],
        ["Intercept", "ball_count[[0,2]]", "strike_count[[0,1]]", "ball_count[[0,2]]:strike_count[[0,1]]"]
    ])).reset_index()
results.columns = ["component", "param", "_value", "pvalue"]
results = results.pivot(index="param", columns="component", values=["_value", "pvalue"]).T.reset_index()
results.columns = ["stat", "component", "Intercept", "b[0,2]",
                   "s[0,1]", "b[0,2]:s[0,1]"]
results = results.pivot(index="component", columns="stat")


params.index = pd.MultiIndex.from_product([
        ["c"+str(i) for i in range(10)],
        ["Intercept", "ball_count[[0,2]]", "strike_count[[0,1]]", "ball_count[[0,2]]:strike_count[[0,1]]"]
    ])
params = params.reset_index()
params.columns = ["component", "param", "value"]
param_mat = params.pivot(index="component", columns="param", values="value").T


pvalues.index = pd.MultiIndex.from_product([
        ["c"+str(i) for i in range(10)],
        ["Intercept", "ball_count[[0,2]]", "strike_count[[0,1]]", "ball_count[[0,2]]:strike_count[[0,1]]"]
    ])
pvalues = pvalues.reset_index()
pvalues.columns = ["component", "param", "value"]
pvalue_mat = pvalues.pivot(index="component", columns="param", values="value").T






fig = plt.figure(figsize=(8, 3.8))
param_mat[(pvalue_mat > 0.05 / 3)] = np.nan
plt.imshow(param_mat, cmap=newcmp, vmin=-0.4, vmax=0.4)
cbar = plt.colorbar()
plt.grid(False)
plt.xticks(range(10), components_names, rotation=45)
plt.yticks(range(4), [
    "Count: 3 balls, 2 strikes",
    "Count: [0, 2] balls, 2 strikes",
    "Count: 3 balls, [0, 1] strikes",
    "Count: [0, 2] balls, [0, 1] strikes"])
plt.title("LMM: Significant Effects",
		  loc="left",fontweight="bold")
fig.tight_layout()

plt.show()

plt.savefig("./fig/lmm_effects.pdf")












lmm = smf.mixedlm(
    "value ~ 0 + ball_count * strike_count * component",
    data=df_long,
    groups=df_long["umpire"]
)
lmm = lmm.fit()


results = pd.DataFrame(
    {"params": params.to_numpy(), "pvalues": pvalues.to_numpy()},
    index=pd.MultiIndex.from_product([
        ["c"+str(i) for i in range(10)],
        ["b(2,3]", "b[0,2]", "s[0,1]", "b[[0,2]:s[0,1]"]
    ])).reset_index()
results.columns = ["component", "param", "_value", "pvalue"]
results = results.pivot(index="param", columns="component", values=["_value", "pvalue"]).T.reset_index()
results.columns = ["stat", "component", "b(2,3]:s(1,2]", "b[0,2]:s(1,2]", "b(2,3]:s[0,1]", "b[[0,2]:s[0,1]"]
results = results.pivot(index="component", columns="stat")

print(results.applymap("{0:.4f}".format))




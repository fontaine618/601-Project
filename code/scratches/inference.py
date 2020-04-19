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

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

plt.style.use("seaborn")
sys.path.extend(['/home/simon/Documents/601-Project/code'])
encoder_path = "./data/models/encoding/umpire_balls_strikes_fit.txt"

with open(encoder_path, "rb") as f:
    _, embeddings, groups, _, _ = pickle.load(f)

df = pd.DataFrame(embeddings, index=pd.MultiIndex.from_tuples(groups)).reset_index()
df.columns = ["umpire", "ball_count", "strike_count", *["c"+str(i) for i in range(10)]]
df["ball_count"] = df["ball_count"].str.replace("b_count_", "")
df["strike_count"] = df["strike_count"].str.replace("s_count_", "")

X = df[["umpire", "ball_count", "strike_count"]].to_numpy()
y = df[["c"+str(i) for i in range(10)]].to_numpy()

manova = MANOVA.from_formula("c0+c1+c2+c3+c4+c5+c6+c7+c8+c9~umpire+ball_count*strike_count", data=df)
print(manova.mv_test())

anova = ols("c9~umpire+ball_count*strike_count", data=df).fit()
print(anova.summary())

# LMM
df_long = df.melt(
    id_vars=["umpire", "ball_count", "strike_count"],
    value_vars=["c"+str(i) for i in range(10)],
    var_name="component",
    value_name="value"
)

X = df_long[["umpire", "ball_count", "strike_count", "component"]].to_numpy()
y = df_long["value"].to_numpy()

lmm = smf.mixedlm(
    "value ~ 1 + ball_count * strike_count * component",
    data=df_long,
    groups="umpire",
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



N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(0/256, 255/256, N)
vals[:, 1] = np.linspace(39/256, 203/256, N)
vals[:, 2] = np.linspace(76/256, 2/256, N)
newcmp = matplotlib.colors.ListedColormap(vals)
newcmp.set_bad(color='white')


fig = plt.figure(figsize=(8, 3.8))
param_mat[(pvalue_mat > 0.05)] = np.nan
plt.imshow(param_mat, cmap=newcmp)
cbar = plt.colorbar()
plt.grid(False)
plt.xticks(range(10), [
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
], rotation=45)
plt.yticks(range(4), [
    "Count: 3 balls, 2 strikes",
    "Count: [0, 2] balls, 2 strikes",
    "Count: 3 balls, [0, 1] strikes",
    "Count: [0, 2] balls, [0, 1] strikes"])
plt.title("LMM: Significant Effects",
		  loc="left",fontweight="bold")
fig.tight_layout()
plt.savefig("./fig/lmm_effects.pdf")

plt.show()
cols = ["#00274c", "#ffcb05"]









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




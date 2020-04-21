import sys
from data.pitchfx import PitchFxDataset
import pandas as pd
from tables.utils import add_header, change_fontsize, add_divider

sys.path.extend(['/home/simon/Documents/601-Project/code'])

pitchfx = PitchFxDataset()

pitches = pitchfx.pitchfx

# ---------------- compute experiements --------------------

# count
pitches["ball_bin"] = pd.cut(
    x=pitches["b_count"],
    bins=[0, 2, 3],
    labels=["[0,2]", "{3}"],
    include_lowest=True
)
pitches["strike_bin"] = pd.cut(
    x=pitches["s_count"],
    bins=[0, 1, 2],
    labels=["[0,1]", "{2}"],
    include_lowest=True
)

# movement
pitches["move_horiz"] = pd.cut(
    x=pitches["pfx_x_std"],
    bins=[-60, 0, 60],
    labels=["Inward", "Outward"],
    include_lowest=True
)
pitches["move_vert"] = pd.cut(
    x=pitches["pfx_z"],
    bins=[-20, 5, 20],
    labels=["Upward", "Downward"],
    include_lowest=True
)

# inning
pitches["inning_bin"] = pd.cut(
    x=pitches["inning"],
    bins=[1, 6, 18],
    labels=["[1,6]", "7+"],
    include_lowest=True
)




columns = [
    "type",
    "stand",
    "p_throws",
    "umpire_HP",
    "type_from_sz",
    "ball_bin",
    "strike_bin",
    "move_horiz",
    "move_vert",
    "inning_bin"
]

index = []
for col in columns:
    index.extend([
        (col, val) for val in sorted(pitches[col].unique())
    ])
index = pd.MultiIndex.from_tuples(index, names=["var", "value"])

summary = pd.DataFrame(columns=["count", "freq"], index=index)

n = len(pitches)

for var, val in list(index):
    nv = sum(pitches[var] == val)
    summary.loc[(var, val)] = [nv, nv/n]

summary.reset_index(inplace=True)
desc = {
    "type": f"Umpire's call ({len(pitches['type'].unique())})",
    "stand": f"Batter's stand ({len(pitches['stand'].unique())})",
    "p_throws": f"Pitcher's throwing arm ({len(pitches['p_throws'].unique())})",
    "umpire_HP": f"Home plate umpire ({len(pitches['umpire_HP'].unique())})",
    "type_from_sz": f"Call from official strike zone ({len(pitches['type_from_sz'].unique())})",
    "ball_bin": f"Ball count bins ({len(pitches['ball_bin'].unique())})",
    "strike_bin": f"Strike count bins ({len(pitches['strike_bin'].unique())})",
    "move_horiz": f"Horizontal movement bins ({len(pitches['move_horiz'].unique())})",
    "move_vert": f"Vertical movement bins ({len(pitches['move_vert'].unique())})",
    "inning_bin": f"Inning bins ({len(pitches['inning_bin'].unique())})"
}
summary["var"].replace(desc, inplace=True)

summary["freq"] *= 100
summary["freq"] = summary["freq"].apply("{:.2f} %".format)
summary["count"] = summary["count"].apply("{}".format)

summary.drop(index=range(9, 41), inplace=True)


summary.iloc[9] = ["Home plate umpire (39)", "...", "...", "..."]

summary.index = pd.MultiIndex.from_frame(
    summary[["var", "value"]],
    names=["Description (nb. levels)", "Level"]
)
summary.drop(columns=["var", "value"], inplace=True)
summary.columns = ["Count", "Frequency"]

table = summary.to_latex(
    buf=None,
    index=True,
    column_format=r"llrr",
    caption="Description of the categorical variables used in the analysis "
            "after pre-processing and tidying.",
    label="tab:data.desc.cat"
)
print(table)

table = table.replace(
        "&       &   Count & Frequency \\\\\nDescription (nb. levels) & Level &         &",
        "Description (nb. levels) & Level &   Count & Frequency"
    )

table = add_header(table, "Categorical variables description", len(summary.columns))
table = change_fontsize(table, "\\small")
table = add_divider(table, "\\midrule", 4, "Original variables")
table = add_divider(table, "2.38 \% \\\\", 4, "Calculated variables")

print(table)
with open("/home/simon/Documents/601-Project/"
          "tex/report/tables/data_description_categorical.tex", "w") as f:
    f.write(table)
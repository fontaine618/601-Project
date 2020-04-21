import sys
from data.pitchfx import PitchFxDataset
import pandas as pd
from tables.utils import add_header, change_fontsize, add_divider

sys.path.extend(['/home/simon/Documents/601-Project/code'])

pitchfx = PitchFxDataset()

pd.options.display.max_colwidth = 10000

summary = pd.DataFrame(columns=[
    "split",
    "count", "min", "med", "max"
])
# ---------------- compute experiements --------------------

# count
df = pitchfx.group_by(
    umpire_HP="all",
    b_count=[0, 2, 3],
    s_count=[0, 1, 2]
)
counts = df.agg("count")["px"]
stats = counts.agg(["count", "min", "median", "max"]).to_numpy()
summary.loc[1] = [
    "Umpire (39),\newline Ball count ([0,2], {3}),\newline Strike count ([0,1], {2})",
    *stats
]
# movement
df = pitchfx.group_by(
	umpire_HP="all",
	pfx_x=[-60, 0, 60],
	pfx_z=[-20, 5, 20]
)
counts = df.agg("count")["px"]
stats = counts.agg(["count", "min", "median", "max"]).to_numpy()
summary.loc[2] = [
    "Umpire (39),\newline Horiz. movement (inward, outward),\newline Vert. movement (upward, downward)",
    *stats
]
# batter/pitcher
df = pitchfx.group_by(
	umpire_HP="all",
	p_throws="all",
    stand="all"
)
counts = df.agg("count")["px"]
stats = counts.agg(["count", "min", "median", "max"]).to_numpy()
summary.loc[3] = [
    "Umpire (39),\newline Pitcher's arm (L, R),\newline Batter's stand (L, R)",
    *stats
]
# score inning
df = pitchfx.group_by(
	umpire_HP="all",
	score_diff_b_p=[-25, -2, 1, 25],
	inning=[1, 6, 18]
)
counts = df.agg("count")["px"]
stats = counts.agg(["count", "min", "median", "max"]).to_numpy()
summary.loc[4] = [
    "Umpire (39),\newline Batter's score diff. ($<$-1, [-1, +1], $>$+1),\newline Inning ([1,6], 7+)",
    *stats
]

summary.columns = [
    "Splittings (levels)",
    "Count", "Min.", "Med.", "Max."
]

table = summary.to_latex(
    buf=None,
    index=True,
    float_format="{:.0f}".format,
    column_format="lp{6.5cm}rrrr",
    caption="Description of the experiments conducted.",
    label="tab:data.exps",
    escape=False
)
print(table)

table = add_header(table, "Experiments description", len(summary.columns)+1)
table = change_fontsize(table, "\\small")

table = table.replace(
    "\\\\\\addlinespace\n\n{}",
    "\\\\\\addlinespace\n & & & \\multicolumn{3}{c}{Sample sizes}\\\\\cmidrule(l){4-6}\n{}",
)

for i in range(2, 5):
    table = table.replace(
        f"\\\\\n{i}",
        f"\\\\\\addlinespace\n{i}"
    )

print(table)
with open("/home/simon/Documents/601-Project/"
          "tex/report/tables/experiments.tex", "w") as f:
    f.write(table)
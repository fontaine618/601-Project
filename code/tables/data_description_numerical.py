import sys
from data.pitchfx import PitchFxDataset
import pandas as pd
from tables.utils import add_header, change_fontsize, add_divider

sys.path.extend(['/home/simon/Documents/601-Project/code'])

pitchfx = PitchFxDataset()

pitches = pitchfx.pitchfx.reset_index()

summary = pitches[[
    "px",
    "pz",
    "sz_bot",
    "sz_top",
    "b_count",
    "s_count",
    "b_score",
    "p_score",
    "inning",
    "pfx_x",
    "pfx_z",
    "break_angle",
    "px_std",
    "pz_std",
    "pfx_x_std",
    "score_diff_b_p",
]].describe(
    percentiles=[0.5]
).T

summary["desc"] = [
    "Horiz. pitch location (ft)",
    "Vert. pitch location (ft)",
    "Strike zone lower limit (ft)",
    "Strike zone upper limit (ft)",
    "Balls in the count",
    "Strikes in the count",
    "Batter's team score",
    "Pitcher's team score",
    "Current inning",
    "Horizontal pitch movement (ft)",
    "Vertical pitch movement (ft)",
    "Pitch break angle",
    "Std horiz. pitch location (ft)",
    "Std vert. pitch location (ft)",
    "Std horiz. pitch movement (ft)",
    "Score diff. (batter - pitcher)"
]
summary["type"] = ["Continuous"]*4 + ["Discrete"]*5 + ["Continuous"]*6 + ["Discrete"]

summary = summary[["desc", "type", "mean", "std", "min", "50%", "max"]]
summary.columns = [
    "Description", "Type",
    "Mean", "Std dev.", "Min.",
    "Med.", "Max."
]

table = summary.to_latex(
    buf=None,
    float_format="{:.3f}".format,
    index=False,
    column_format=r"llrrrrrrr",
    caption="Description of the numerical variables used in the analysis "
            "after pre-processing and tidying.",
    label="tab:data.desc.num"
)

table = add_header(table, "Numerical variables description", len(summary.columns))
table = change_fontsize(table, "\\small")
table = add_divider(table, "\\midrule", len(summary.columns), "Original variables")
table = add_divider(table, "63.900 \\\\", len(summary.columns), "Calculated variables")
print(table)
with open("/home/simon/Documents/601-Project/"
          "tex/report/tables/data_description_numerical.tex", "w") as f:
    f.write(table)
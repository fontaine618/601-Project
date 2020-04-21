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


# --------------------- score innings ---------------------

pitchfx.pitchfx["score_diff_b_p"] = pitchfx.pitchfx["b_score"] - pitchfx.pitchfx["p_score"]

df_score_inning = pitchfx.group_by(
	umpire_HP="all",
	score_diff_b_p=[-25, -2, 1, 25],
	inning=[1, 6, 18]
)

ids_score_inning = [groups.index(gr) for gr, _ in df_score_inning if gr in groups]

embeddings_score_inning = embeddings[ids_score_inning, :]
groups_score_inning = [groups[i] for i in ids_score_inning]

df_score_inning = pd.DataFrame(embeddings_score_inning, index=pd.MultiIndex.from_tuples(groups_score_inning)).reset_index()
df_score_inning.columns = ["umpire", "score", "inning", *["c"+str(i) for i in range(10)]]

umpires_score_inning = pd.DataFrame(
    {
        components_names[i]:
        ols("c" + str(i) + "~umpire+score*inning", data=df_score_inning).fit().params[:-5]
        for i in range(10)
    }
)
# umpires_score_inning.iloc[1:] = umpires_score_inning.iloc[1:] + umpires_score_inning.iloc[0]
umpires_score_inning = umpires_score_inning.iloc[1:]
# --------------------- pitcher batter ---------------------

df_pitcher_batter = pitchfx.group_by(
	umpire_HP="all",
	p_throws="all",
    stand="all"
)

ids_pitcher_batter = [groups.index(gr) for gr, _ in df_pitcher_batter if gr in groups]

embeddings_pitcher_batter = embeddings[ids_pitcher_batter, :]
groups_pitcher_batter = [groups[i] for i in ids_pitcher_batter]

df_pitcher_batter = pd.DataFrame(embeddings_pitcher_batter, index=pd.MultiIndex.from_tuples(groups_pitcher_batter)).reset_index()
df_pitcher_batter.columns = ["umpire", "pitcher", "batter", *["c"+str(i) for i in range(10)]]

umpires_pitcher_batter = pd.DataFrame(
    {
        components_names[i]:
        ols("c" + str(i) + "~umpire+pitcher*batter", data=df_pitcher_batter).fit().params[:-3]
        for i in range(10)
    }
)
# umpires_pitcher_batter.iloc[1:] = umpires_pitcher_batter.iloc[1:] + umpires_pitcher_batter.iloc[0]
umpires_pitcher_batter = umpires_pitcher_batter.iloc[1:]
# --------------------- count ---------------------

df_ball_strike = pitchfx.group_by(
    umpire_HP="all",
    b_count=[0, 2, 3],
    s_count=[0, 1, 2]
)

ids_ball_strike = [groups.index(gr) for gr, _ in df_ball_strike if gr in groups]

embeddings_ball_strike = embeddings[ids_ball_strike, :]
groups_ball_strike = [groups[i] for i in ids_ball_strike]

df_ball_strike = pd.DataFrame(embeddings_ball_strike, index=pd.MultiIndex.from_tuples(groups_ball_strike)).reset_index()
df_ball_strike.columns = ["umpire", "ball_count", "strike_count", *["c"+str(i) for i in range(10)]]
df_ball_strike["ball_count"] = df_ball_strike["ball_count"].str.replace("b_count_", "")
df_ball_strike["strike_count"] = df_ball_strike["strike_count"].str.replace("s_count_", "")

umpires_ball_strike = pd.DataFrame(
    {
        components_names[i]:
        ols("c" + str(i) + "~umpire + ball_count*strike_count", data=df_ball_strike).fit().params[:-3]
        for i in range(10)
    }
)
# umpires_ball_strike.iloc[1:] = umpires_ball_strike.iloc[1:] + umpires_ball_strike.iloc[0]
umpires_ball_strike = umpires_ball_strike.iloc[1:]
# ----------- MERGE --------------

umpire_intercepts = np.array([
    umpires_ball_strike.to_numpy(),
    umpires_pitcher_batter.to_numpy(),
    umpires_score_inning.to_numpy(),
    umpires_movement.to_numpy()
])

centered = umpire_intercepts - np.expand_dims(umpire_intercepts.mean(1), 1)

z = centered / np.expand_dims(umpire_intercepts.std(1), 1)

scores = pd.Series((z ** 2).sum(2).sum(0), index=umpires_ball_strike.index)
scores.sort_values()
# score difference and inning 7+

pitchfx.pitchfx["score_diff_b_p"] = pitchfx.pitchfx["b_score"] - pitchfx.pitchfx["p_score"]

df = pitchfx.group_by(
	umpire_HP="all",
	score_diff_b_p=[-25, -2, 1, 25],
	inning=[1, 6, 18]
)
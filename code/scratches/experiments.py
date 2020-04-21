pitchfx = PitchFxDataset()

# Balls and strike counts

df = pitchfx.group_by(
    umpire_HP="all",
    b_count=[0, 2, 3],
    s_count=[0, 1, 2]
)

# LR movement U/D movement
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

# LR pitchers vs LR batters
df = pitchfx.group_by(
	umpire_HP="all",
	p_throws="all",
    stand="all"
)
# score difference and inning 7+

pitchfx.pitchfx["score_diff_b_p"] = pitchfx.pitchfx["b_score"] - pitchfx.pitchfx["p_score"]

df = pitchfx.group_by(
	umpire_HP="all",
	score_diff_b_p=[-25, -2, 1, 25],
	inning=[1, 6, 18]
)
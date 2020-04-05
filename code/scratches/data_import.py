import sys
sys.path.extend(['/home/simon/Documents/601-Project/code'])
from data import PitchFxDataset
import matplotlib.pyplot as plt

pitchfx = PitchFxDataset()

df = pitchfx.group_by(
    umpire_HP="all",
    stand="all",
    b_count=[0, 2, 3],
    s_count=[0, 1, 2]
)

# to iterate through all:
for levels, d in df:
    print(len(d), levels)

plt.scatter(pitchfx.pitchfx["pz"][:1000], pitchfx.pitchfx["pz_std"][:1000])
plt.hist(pitchfx.pitchfx["pz_std"] - pitchfx.pitchfx["pz"])
plt.show()
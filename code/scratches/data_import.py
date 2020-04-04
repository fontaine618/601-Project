import sys
sys.path.extend(['/home/simon/Documents/601-Project/code'])
from data import PitchFxDataset
import matplotlib.pyplot as plt

pitchfx = PitchFxDataset()

request = {"umpire_HP": "all", "stand": "all", "start_speed": [0, 80, 90, 150]}

df = pitchfx.group_by(
    umpire_HP="all",
    stand="all",
    start_speed=[0, 80, 90, 150]
)

# to iterate through all:
for levels, d in df:
    print(levels)
    print(d)
    break

plt.scatter(pitchfx.pitchfx["pz"][:1000], pitchfx.pitchfx["pz_std"][:1000])
plt.hist(pitchfx.pitchfx["pz_std"] - pitchfx.pitchfx["pz"])
plt.show()
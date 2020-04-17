import sys
sys.path.extend(['/home/simon/Documents/601-Project/code'])
from data.pitchfx import PitchFxDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pitchfx = PitchFxDataset()
pd.crosstab(pitchfx.pitchfx["type"], pitchfx.pitchfx["type_from_sz"])
df = pitchfx.group_by(
    umpire_HP="all",
    stand="all",
)

# to iterate through all:
for levels, d in df:
    print(len(d), levels)

plt.scatter(pitchfx.pitchfx["pz"][:1000], pitchfx.pitchfx["pz_std"][:1000])
plt.hist(pitchfx.pitchfx["pz_std"] - pitchfx.pitchfx["pz"])
plt.show()


from spyre import server
from plot.utils import batter_outline, strike_zone
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

sys.path.extend(['/home/simon/Documents/601-Project/code'])
encoder_path = "./data/models/encoding/PCA10_umpire_balls_strikes.txt"

with open(encoder_path, "rb") as f:
    encoder, embeddings, x_range, y_range = pickle.load(f)

min = embeddings.min(0)
max = embeddings.max(0)
n_components = encoder.encoder.n_components
plt.style.use("seaborn")


class DecodingApp(server.Launch):

    title = "Embedding decoder to strike zone"

    outputs = [
        {
            "type": "plot",
            "id": "plot"
        }
    ]

    inputs = [
        {
            "type": "slider",
            "key": "c" + str(i+1),
            "label": "Component " + str(i + 1),
            "value": 0,
            "min": round(min_),
            "max": round(max_),
            "step": (round(max_) - round(min_)) / 10.,
            "action_id": "plot"
        }
        for i, min_, max_ in zip(range(n_components), min, max)
    ]

    def getPlot(self, params):
        u = np.array([[float(params["c" + str(i+1)]) for i in range(n_components)]])
        sz = encoder.inverse_transform(u, ["sz"])["sz"].clip(0, 1)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(4 * sz * (1. - sz), extent=(*x_range, *y_range[::-1]), alpha=(4 * sz * (1. - sz)).astype(float))
        ax.plot(*batter_outline(), scalex=False, scaley=False, color="black")
        ax.plot(*strike_zone(), scalex=False, scaley=False, color="white", linewidth=1, linestyle="--")
        return fig


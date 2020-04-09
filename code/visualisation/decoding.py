from spyre import server
from plot.utils import batter_outline, strike_zone
import numpy as np
import matplotlib.pyplot as plt


class DecodingApp(server.Launch):

    def __init__(self, n_components, encoder, min, max, x_range, y_range):
        self.n_components = n_components
        self.encoder = encoder
        self.x_range = x_range
        self.y_range = y_range
        self.title = "Embedding decoder to strike zone"
        self.outputs = [
            {
                "type": "plot",
                "id": "plot"
            }
        ]
        self.inputs = [
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
        u = np.array([[float(params["c" + str(i+1)]) for i in range(self.n_components)]])
        sz = self.encoder.inverse_transform(u, ["sz"])["sz"].clip(0, 1)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(4 * sz * (1. - sz), extent=(*self.x_range, *self.y_range[::-1]), alpha=(4 * sz * (1. - sz)).astype(float))
        ax.plot(*batter_outline(), scalex=False, scaley=False, color="black")
        ax.plot(*strike_zone(), scalex=False, scaley=False, color="white", linewidth=1, linestyle="--")
        return fig


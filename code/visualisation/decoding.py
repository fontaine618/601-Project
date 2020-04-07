from spyre import server
from plot.utils import batter_outline, strike_zone
import numpy as np
import matplotlib.pyplot as plt


class DecondingApp(server.App):

    title = "Embedding decoder to strike zone"

    outputs = [
        {
            "type": "plot",
            "id": "sz_plot"
        }
    ]

    def __init__(self, encoder, embeddings):
        plt.style.use("seaborn")
        self.encoder = encoder
        self.embeddings = embeddings
        self.n_components = encoder.encoder.n_components
        min = embeddings.min(0)
        max = embeddings.max(0)
        mean = embeddings.mean(0)
        self.inputs = [
            {
                "type": "slider",
                "key": i,
                "label": "Component " + str(i + 1),
                "value": mean_,
                "min": min_,
                "max": max_,
                "step": (max_ - min_) / 100.,
                "action_id": "sz_plot"
            }
            for i, min_, max_, mean_ in zip(range(self.n_components), min, max, mean)
        ]

    def getPlot(self, params):
        u = np.array([[params[i] for i in range(self.n_components)]])
        sz = self.encoder.inverse_transform(u, ["sz"])["sz"].clip(0, 1)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(4 * sz * (1. - sz), extent=(-2, 2, 1.5, 5), alpha=(4 * sz * (1. - sz)).astype(float))
        ax.plot(*batter_outline(), scalex=False, scaley=False, color="black")
        ax.plot(*strike_zone(), scalex=False, scaley=False, color="white", linewidth=1, linestyle="--")
        return fig


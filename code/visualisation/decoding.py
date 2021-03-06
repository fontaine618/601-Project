from spyre import server
from plot.utils import batter_outline, strike_zone
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DecodingApp(server.Launch):

    def __init__(self, encoder_path):
        with open(encoder_path, "rb") as f:
            encoder, embeddings, _, x_range, y_range = pickle.load(f)

        min = embeddings.min(0)
        max = embeddings.max(0)

        self.n_components = encoder.n_components
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
                "min": min_,
                "max": max_,
                "step": (max_ - min_) / 100.,
                "action_id": "plot"
            }
            for i, min_, max_ in zip(range(self.n_components), min, max)
        ]

    def getPlot(self, params):
        u = np.array([[float(params["c" + str(i+1)]) for i in range(self.n_components)]])
        sz = self.encoder.inverse_transform(u.reshape((1, -1))).clip(0, 1).reshape((100, 100))
        sz_alpha = (4 * sz * (1. - sz)).astype(float)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(
            sz,
            extent=(*self.x_range, *self.y_range[::-1]),
            # alpha=sz
        )
        ax.plot(*batter_outline(), scalex=False, scaley=False, color="black")
        ax.plot(*strike_zone(), scalex=False, scaley=False, color="white", linewidth=1, linestyle="--")
        return fig


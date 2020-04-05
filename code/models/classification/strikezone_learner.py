import pandas as pd
import numpy as np
import copy


class StrikezoneLearner:

    def __init__(self, df, classifier, x_range=(-1.5, 1.5), y_range=(4, 1), res=100):
        self.classifier = classifier
        if "fit" not in dir(self.classifier):
            raise NotImplementedError("classifier must implement fit")
        if "predict_proba" not in dir(self.classifier):
            raise NotImplementedError("classifier must implement predict")
        self.grouped_pitches = df
        self.groups = self.grouped_pitches.agg({
            "px": ["count"]
        })
        self.groups.columns = ["count"]
        self.groups["model"] = None
        self.groups["strikezone"] = None
        self._fitted = False
        self.x_range = x_range
        self.y_range = y_range
        self.res = res
        self.grid_x, self.grid_y = np.meshgrid(
            np.linspace(*self.x_range, num=self.res),
            np.linspace(*self.y_range, num=self.res)
        )
        self.X_sz = np.concatenate([self.grid_x.reshape((-1, 1)), self.grid_y.reshape((-1, 1))], axis=1)

    def fit(self, levels, pitches):
        self.groups.at[levels, "model"] = copy.deepcopy(self.classifier.fit(
            pitches[["px_std", "pz_std"]].to_numpy(),
            pitches[["type"]].to_numpy().reshape((-1))
        ))

    def fit_all(self):
        for levels, pitches in self.grouped_pitches:
            self.fit(levels, pitches)
        self._fitted = True

    def predict_strikezone(self, levels, model):
        self.groups.at[levels, "strikezone"] = model.predict_proba(self.X_sz).reshape((self.res, self.res))

    def predict_strikezone_all(self):
        if not self._fitted:
            raise ValueError("models are not fitted yet; run .fit_all() first!")
        for levels, model in self.groups["model"].items():
            self.predict_strikezone(levels, model)



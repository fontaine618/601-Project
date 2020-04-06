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
        self.pitches = df
        # self.groups = self.pitches.agg({
        #     "px": ["count"]
        # })
        # self.groups.columns = ["count"]
        # self.groups["model"] = None
        # self.groups["strikezone"] = None

        self.counts = self.pitches.agg({
            "px": ["count"]
        }).to_dict()[("px", "count")]
        self.fit = dict()
        self.strikezone = dict()

        self.x_range = x_range
        self.y_range = y_range
        self.res = res
        self.grid_x, self.grid_y = np.meshgrid(
            np.linspace(*self.x_range, num=self.res),
            np.linspace(*self.y_range, num=self.res)
        )
        self.X_sz = np.concatenate([self.grid_x.reshape((-1, 1)), self.grid_y.reshape((-1, 1))], axis=1)

    def _fit_one(self, pitches):
        return self.classifier.fit(
            pitches[["px_std", "pz_std"]].to_numpy(),
            pitches[["type"]].to_numpy().reshape((-1))
        )

    def fit_all(self):
        for levels, pitches in self.pitches:
            self.fit[levels] = copy.deepcopy(self._fit_one(pitches))
            break


    def _predict_strikezone(self, levels):
        if levels in self.fit:
            fit = self.fit[levels]
        else:
            pitches = self.pitches.get_group(levels)
            fit = self.classifier.fit(
                pitches[["px_std", "pz_std"]].to_numpy(),
                pitches[["type"]].to_numpy().reshape((-1))
            )
        return fit.predict_proba(self.X_sz).reshape((self.res, self.res))

    def predict_strikezone_all(self):
        for levels, pitches in self.pitches:
            self.strikezone[levels] = self._predict_strikezone(levels)
            break




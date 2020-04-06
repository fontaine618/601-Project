import numpy as np
import copy
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score


class StrikezoneLearner:

    def __init__(self, df, classifier, x_range=(-1.5, 1.5), y_range=(4, 1), res=100):
        self.classifier = classifier
        if "fit" not in dir(self.classifier):
            raise NotImplementedError("classifier must implement fit")
        if "predict_proba" not in dir(self.classifier):
            raise NotImplementedError("classifier must implement predict")
        self.pitches = df
        self.counts = self.pitches.agg({
            "px": ["count"]
        }).to_dict()[("px", "count")]
        self.fit = dict()
        self.strikezone = dict()
        self.cv_accuracy = dict()

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
            pitches[["type_01"]].to_numpy().reshape((-1))
        )

    def fit_all(self):
        for levels, pitches in self.pitches:
            self.fit[levels] = copy.deepcopy(self._fit_one(pitches))

    def _predict_strikezone(self, levels):
        if levels in self.fit:
            fit = self.fit[levels]
        else:
            pitches = self.pitches.get_group(levels)
            fit = self.classifier.fit(
                pitches[["px_std", "pz_std"]].to_numpy(),
                pitches[["type_01"]].to_numpy().reshape((-1))
            )
        # pred = fit.predict_proba(self.X_sz)
        pred = fit.predict(self.X_sz)
        if len(pred.shape) == 2:
            pred = pred[:, 1]
        return pred.reshape((self.res, self.res))

    def predict_strikezone_all(self):
        for levels, pitches in self.pitches:
            self.strikezone[levels] = self._predict_strikezone(levels)

    def _cv_one(self, levels, pitches, n_folds=5):
        accuracy = np.zeros(n_folds)
        n = len(pitches)
        fold_id = np.array([[i]*(n//n_folds + 1) for i in range(n_folds)]).ravel()
        np.random.seed(1)
        np.random.shuffle(fold_id)
        fold_id = fold_id[:n]
        for i in range(n_folds):
            train = pitches.iloc[fold_id != i]
            test = pitches.iloc[fold_id == i]
            self.classifier.fit(
                train[["px_std", "pz_std"]].to_numpy(),
                train[["type_01"]].to_numpy().reshape((-1))
            )
            pred = self.classifier.predict(
                test[["px_std", "pz_std"]].to_numpy()
            )
            accuracy[i] = accuracy_score(test[["type_01"]].to_numpy().reshape((-1)), pred)
        return levels, accuracy.mean()

    def _cv_one_scikit(self, levels, pitches, n_folds=5, scoring="accuracy"):
        accuracy = cross_val_score(
            self.classifier,
            pitches[["px_std", "pz_std"]].to_numpy(),
            pitches[["type_01"]].to_numpy().reshape((-1)),
            cv=n_folds,
            scoring=scoring
        )
        return levels, accuracy.mean()

    def cv_all(self, n_folds=5, n_jobs=-1, prefer="threads", scoring="accuracy"):
        out = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(self._cv_one_scikit)(levels, pitches, n_folds, scoring)
            # delayed(self._cv_one)(levels, pitches, n_folds)
            for levels, pitches in self.pitches
        )
        self.cv_accuracy = dict((levels, acc) for levels, acc in out)
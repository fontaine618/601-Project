import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from time import time
from collections import defaultdict
from functools import partial


# TODO: improve error checking.
# TODO: The method to pass tuples of tuples in case we want CV is clumsy
# TODO: Probably unnecessary to save the train probabilities
# TODO: I always assume that the outer loop should be paralleled but maybe that's not always best?

class StrikeZoneLearner:

    def __init__(self, scoring="accuracy"):
        # TODO: I think at initialisation, the fitter should be
        # TODO: agnostic of the data and algorithmic parameters
        # Used to store the fitting results, keys: groups,
        # values, tuple of probs + best classifier.
        self.fits = defaultdict(object)
        self.probabilities = defaultdict(partial(np.ndarray, 0))
        self.scores = defaultdict(float)
        self.groups = set()
        self.scoring = scoring
        self.strike_zones = defaultdict(partial(np.ndarray, 0))
        self.groups_processed = 0
        self.n_groups = 0
        self.time0 = time()
        self.params = defaultdict(dict)

    def fit(self, pitches, labels, classifier, param_grid=None,
            cv=False, n_jobs=-1, cv_folds=None, group="All",
            single_call=True):
        # logging
        if self.groups_processed == 0:
            self.time0 = time()
        if isinstance(pitches, pd.core.groupby.generic.DataFrameGroupBy):
            raise ValueError("Use fit_groups to fit groups.")
        if cv:
            if param_grid is None:
                raise ValueError("A parameter grid needs to be provided"
                                 " for cross validation")
            gsv = GridSearchCV(classifier, param_grid=param_grid,
                               n_jobs=n_jobs, scoring=self.scoring, cv=cv_folds)
            fit = gsv.fit(pitches, labels)
            score = fit.best_score_
            fit = fit.best_estimator_
        else:
            fit = classifier.fit(pitches, labels)
            score = accuracy_score(fit.predict(pitches), labels)
        # logging
        self.groups_processed += 1
        if self.groups_processed % 10 == 0:
            print(self.groups_processed, "/", self.n_groups, "groups processed in", round(time() - self.time0, 2), "s")
            self.time0 = time()

        # This hack so that fit can be called individually
        # but we can also make use of paralleled processing
        # if this called from the other methods
        if single_call:
            self.fits[group] = fit
            self.params[group] = fit.best_params_
            self.probabilities[group] = fit.predict_proba(pitches)
            self.scores[group] = score
            return self
        else:
            if score > self.scores[group]:
                # found a better classifier
                print("    New best classifier for group", group)
                print("    - Classifier:", type(classifier).__name__)
                print("    - Params:", gsv.best_params_)
                print("    - Score", score, "(previous={})".format(self.scores[group]))
                self.fits[group] = fit
                self.params[group] = fit.best_params_
                self.probabilities[group] = fit.predict_proba(pitches)
                self.scores[group] = score
            return self

    def fit_groups(self, df, data_col, label_col, classifier=None,
                   param_grid=None, cv=False, cv_folds=None, n_jobs=-1,
                   prefer=None):
        # logging
        print(type(classifier).__name__, "=" * 80)
        for k, v in param_grid.items():
            print(k, ":", v)
        time0 = time()
        self.n_groups = len(df)
        self.groups_processed = 0

        # This expects df to be an instance of
        # pd.core.groupby.generic.DataFrameGroupBy
        if not isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
            raise ValueError("fit_groups expects a grouped DataFrame")

        # This should be the case if we want to fit
        # one classifier to all groups
        if cv:
            n_jobs_ = n_jobs
            n_jobs = 1
        else:
            n_jobs_ = 1
        Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(self.fit)(pitches[data_col].to_numpy(),
                              pitches[label_col].to_numpy().reshape((-1)),
                              classifier, param_grid, cv, n_jobs=n_jobs_,
                              cv_folds=cv_folds, group=group,
                              single_call=False) for group, pitches in df
        )
        print(type(classifier).__name__, "completed in", time() - time0, "s")
        return self

    def fit_groups_all_classifiers(self, df, data_col, label_col,
                                   classifiers, cv=False, cv_folds=None, n_jobs=-1):
        # This expects df to be an instance of
        # pd.core.groupby.generic.DataFrameGroupBy
        if not isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
            raise ValueError("fit_groups expects a grouped DataFrame")

        # For this, classifiers should be a tuple of 2 tuples,
        # the first entry the classifier object, the second a
        # dictionary with parameters for CV.
        if np.ndim(classifiers) > 2:
            raise ValueError("Classifier argument should either"
                             " be an iterable over Classifier instances"
                             " or a tuple with one entry the classifier "
                             " instance and one entry a dictionary for CV.")
        elif np.ndim(classifiers) == 2:
            cv = True

        # fit on all groups, iterating through classifiers
        for c in classifiers:
            self.fit_groups(df=df, data_col=data_col, label_col=label_col,
                            classifier=c[0], param_grid=c[1], cv=cv,
                            n_jobs=n_jobs, cv_folds=cv_folds)
        return self

    # The instance should be agnostic to the strike zone
    # so that we dont have to retrain in case we want
    # to change the resolution.
    def compute_strike_zones(self, x_range=(-1.5, 1.5), y_range=(4, 1),
                             res=100):
        grid_x, grid_y = np.meshgrid(
            np.linspace(*x_range, num=res),
            np.linspace(*y_range, num=res)
        )
        x_sz = np.concatenate([grid_x.reshape((-1, 1)),
                               grid_y.reshape((-1, 1))], axis=1)
        for group in self.groups:
            pred = self.fits[group].predict_proba(x_sz)
            if len(pred.shape) == 2:
                pred = pred[:, 1]
            self.strike_zones[group] = pred.reshape((res, res))
        return grid_x, grid_y, self.strike_zones

    def plot_results(self, type="bar"):
        if len(self.groups) > 0:
            plt.style.use('seaborn')
            fig, ax = plt.subplots()
            x = []
            y = []
            for i, group in enumerate(self.groups):
                x.append(i)
                y.append(self.scores[group])
            if type == "bar":
                ax.bar(x, y)
            else:
                ax.plot(x, y)
            ax.set_xlabel("Groups")
            ax.set_ylabel("Best scores")
            plt.show()

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# TODO: improve error checking.
# TODO: The method to pass tuples of tuples in case we want CV is clumsy
# TODO: Probably unnecessary to save the train probabilities
# TODO: I always assume that the outer loop should be paralleled but maybe that's not always best?

class StrikeZoneLearner:

    def __init__(self, n_jobs=-1, prefer="threads",
                 scoring="accuracy"):
        # TODO: I think at initialisation, the fitter should be
        # TODO: agnostic of the data and algorithmic parameters
        # Used to store the fitting results, keys: groups,
        # values, tuple of probs + best classifier.
        self.fits = {}
        self.probabilities = {}
        self.scores = {}
        self.groups = []
        self.scoring = scoring
        self.strike_zones = {}

    def fit(self, pitches, labels, classifier, param_grid=None,
            cv=False, n_jobs=-1, cv_folds=None, group="All",
            single_call=True):
        if isinstance(pitches, pd.core.groupby.generic.DataFrameGroupBy):
            raise ValueError("Use fit_groups to fit groups.")
        if cv:
            if param_grid is None:
                raise ValueError("A parameter grid needs to be provided"
                                 " for cross validation")
            gsv = GridSearchCV(classifier, param_grid=param_grid,
                               n_jobs=n_jobs, scoring=self.scoring)
            fit = gsv.fit(pitches, labels)
            score = fit.best_score_
            fit = fit.best_estimator_
        else:
            fit = classifier.fit(pitches, labels)
            score = accuracy_score(fit.predict(pitches), labels)

        # This hack so that fit can be called individually
        # but we can also make use of paralleled processing
        # if this called from the other methods
        if single_call:
            self.fits[group] = fit
            self.probabilities[group] = fit.predict_proba(pitches)
            self.scores[group] = score
            return self
        else:
            return group, fit.predict_proba(pitches), fit, score

    def fit_groups(self, df, data_col, label_col, classifier=None,
                   param_grid=None, cv=False, cv_folds=None, n_jobs=-1,
                   prefer=None, single_classifier=True):
        # This expects df to be an instance of
        # pd.core.groupby.generic.DataFrameGroupBy

        # This should be the case if we want to fit
        # one classifier to all groups
        if cv:
            n_jobs_ = 1
        else:
            n_jobs_ = n_jobs

        out = Parallel(n_jobs=n_jobs, prefer=prefer)(
            delayed(self.fit)(pitches[data_col].to_numpy(),
                              pitches[label_col].to_numpy().reshape((-1)),
                              classifier, param_grid, cv, n_jobs=n_jobs_,
                              cv_folds=cv_folds, group=group,
                              single_call=False) for group, pitches in df
        )

        # Same hack as in fit. Used to that it can be called
        # on its own.
        if single_classifier:
            for group, probabilities, fit, score in out:
                self.fits[group] = fit
                self.probabilities[group] = probabilities
                self.scores[group] = score
            return self
        else:
            return out

    def fit_groups_all_classifiers(self, df, data_col, label_col,
                                   classifiers, cv=False, cv_folds=None, n_jobs=-1,
                                   prefer=None):
        # This expects df to be an instance of
        # pd.core.groupby.generic.DataFrameGroupBy

        if np.ndim(classifiers) > 2:
            raise ValueError("Classifier argument should either"
                             " be an iterable over Classifier instances"
                             " or a tuple with one entry the classifier "
                             " instance and one entry a dictionary for CV.")
        elif np.ndim(classifiers) == 2:
            cv = True

        # For this, classifiers should be a tuple of 2 tuples,
        # the first entry the classifier object, the second a
        # dictionary with parameters for CV.
        if cv:
            out = Parallel(n_jobs=n_jobs, prefer=prefer)(
                delayed(self.fit_groups)(df, data_col, label_col,
                                         c[0], c[1], cv, n_jobs=1,
                                         cv_folds=cv_folds,
                                         single_classifier=False) for c in classifiers
            )
        # Here classifiers should be a list of classifier
        # objects initialised with the necessary params
        # TODO: Maybe this is a bit clumsy but I think
        # TODO: it might be convenient
        else:
            out = Parallel(n_jobs=n_jobs, prefer=prefer)(
                delayed(self.fit_groups)(df, data_col, label_col,
                                         c, cv, n_jobs=1,
                                         cv_folds=cv_folds,
                                         single_classifier=False
                                         ) for c in classifiers
            )

        # TODO: If we use other scoring, than this might not be sensible
        # Necessary since the output might not come in order
        bs = {}
        for group in df.groups:
            bs[group] = -1
        # For every classifier
        for i in range(len(out)):
            # For every group
            for j in range(len(out[i])):
                group = out[i][j][0]
                if out[i][j][3] > bs[group]:
                    self.fits[group] = out[i][j][2]
                    if len(out[i][j][1].shape) == 2:
                        pred = pred[:, 1]
                    else:
                        pred = out[i][j][1]
                    self.probabilities[group] = pred
                    self.scores[group] = out[i][j][3]
        self.groups = df.groups

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
        x_sz = np.concatenate([self.grid_x.reshape((-1, 1)),
                               self.grid_y.reshape((-1, 1))], axis=1)
        for group in self.groups:
            pred = self.fit[group].predict(x_sz)
            if len(pred.shape) == 2:
                pred = pred[:, 1]
            self.strike_zones[group] = pred.reshape((res, res))

    def plot_results(self, type = "bar"):
        if len(self.groups)> 0:
            plt.style.use('seaborn')
            fig, ax = plt.subplots()
            x = []
            y = []
            for group, pitches in self.groups:
                x.append(group)
                y.append(self.scores[group])
            if type == "bar":
                ax.bar(self.groups, self.scores)
            else:
                ax.plot(self.groups, self.scores)
            ax.set_xlabel("Groups")
            ax.set_ylabel("Best scores")
            plt.show()

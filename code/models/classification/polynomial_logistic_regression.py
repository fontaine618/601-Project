"""
This is a module to be used as a reference for building other modules
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted


class PolynomialLogisticRegression(BaseEstimator, ClassifierMixin):
    """ Binary Classifier using Polynomial Logistic Regression
    ----------
    degree : int, default=3
        Degree of the polynomial (with interactions)
    """

    def __init__(self, degree=3,
                 penalty='l2', dual=False, tol=0.0001,
                 C=1.0, fit_intercept=True, intercept_scaling=1,
                 class_weight=None, random_state=None, solver='lbfgs',
                 max_iter=100, multi_class='auto', verbose=0,
                 warm_start=False, n_jobs=None, l1_ratio=None):
        self.degree = degree
        self.penalty = penalty
        self.dual = dual
        self.tol = tol
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.random_state = random_state
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.verbose = verbose
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.l1_ratio = l1_ratio
        self.transform = PolynomialFeatures(degree=degree)
        self.classifier = LogisticRegression(penalty=penalty, dual=dual, tol=tol,
                 C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                 class_weight=class_weight, random_state=random_state, solver=solver,
                 max_iter=max_iter, multi_class=multi_class, verbose=verbose,
                 warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)

    def fit(self, X, y):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X = self.transform.fit_transform(X)
        self.classifier.fit(X, y)
        return self

    def decision_function(self, X):
        X = self.transform.fit_transform(X)
        return self.classifier.decision_function(X)

    def predict(self, X):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = self.transform.fit_transform(X)
        return self.classifier.predict(X)

    def predict_proba(self, X):
        """
        Probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        X = self.transform.fit_transform(X)
        return self.classifier.predict_proba(X)

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        X = self.transform.fit_transform(X)
        return self.classifier.predict_log_proba(X)
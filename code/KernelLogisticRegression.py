"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from scipy.special import expit
from scipy import optimize

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics.pairwise import rbf_kernel



def _kernel_matrix(X, gamma, kernel, Y=None):
    # TODO: Use scikits generic kernel call
    if kernel == "rbf_kernel":
        return rbf_kernel(X, Y=Y, gamma=gamma)
    # TODO: implement
    elif kernel == "polynomial":
        return

    return


def _loss_and_grad(w, K, y, alpha):
    """
    Computes the loss and the gradient
    The loss is the negative likelihood function
    :param w:
        array-like oh shape (n_samples,)
        weights
    :param K:
        array-like of shape (n_sample, n_samples)
        the kernel matrix
    :param y:
        array-like of shape (n_samples,)
        labels
    :param alpha:
        float
        penalty parameter
    :return:
        out : float
            The loss
        grad : ndarray of shape (X.shape[0],)
            The gradient
    """

    n_samples = K.shape[0]

    linear_prediction = K.dot(w)
    penalty = (alpha / 2.) * w.T.dot(K).dot(w)

    # TODO use labels in (-1,1) to improve
    # Loss for kernel logistic regression is the negative likelihood
    out = np.sum(-y * linear_prediction + np.log(1 + np.exp(linear_prediction))) + penalty

    z = expit(linear_prediction)
    z0 = y - z - alpha * w

    grad = -K.dot(z0)

    return out, grad


def _kernel_logistic_regression_path(K, y, tol=1e-4, coef=None,
                                     solver='lbfgs', check_input=True,
                                     C = 1, maxiter=1000):
    """
    Compute the kernel logistic regression model
    :param K:
        array-like of shape (n_sample, n_features)
        Input data
    :param y:
        array-like of shape (n_samples,)
        Input data, target values
    :param tol:
        float, default = 1e-4
        The stopping criterion for the solver
    :param coef:
        array-like of shape (n_samples,)
        Initialisation values of coefficients for the regression
    :param solver:
        str
        The solver to be used
    :param check_input:
        bool, default = True
        Determines whether the input data should be checked
    :return:
        w0 : ndarray of shape
    """

    # TODO: implement
    # if check_input:

    n_samples, n_features = K.shape
    classes = np.unique(y)

    if not classes.size == 2:
        raise ValueError("Only binary Classification.")

    func = _loss_and_grad

    w0 = np.zeros(n_samples, order='F', dtype=K.dtype)

    # TODO: implement other solvers
    if solver == 'lbfgs':
        iprint = [-1, 50, 1, 100, 101]
        opt_res = optimize.minimize(
            func, w0, method="L-BFGS-B", jac=True,
            args=(K, y, 1. / C),
            options={"iprint": iprint, "gtol": tol, "maxiter": maxiter}
        )

        # TODO: Check Result

    w0, loss = opt_res.x, opt_res.fun

    return np.array(w0)


class KernelLogisticRegression(BaseEstimator, ClassifierMixin):
    """ Binary Classifier using Kernel Logistic Regression
    ----------
    kernel : str, default='rbf_kernel'
        Used to determine which kernel function is to be used when generating
        the kernel matrix
    learning_rate : float, default=1
        The learning rate for gradient descent
    gamma : float, default = 1
        Used during the creation for the Gaussian kernel matrix
    C : float, default = 1
        The inverse of the penalty parameter
    """

    def __init__(self,
                 kernel='rbf_kernel',
                 learning_rate=1,
                 gamma=1,
                 C=1,
                 tol=1e-4):
        self.kernel = kernel
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.C = C
        self.tol = tol

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
        if self.C < 0:
            raise ValueError("Penalty must be positive")

        # Necessary for prediction
        # TODO: Find a workaround
        self.X_ = X

        X, y = check_X_y(X, y, accept_sparse=True)

        self.classes_ = np.unique(y)

        K = _kernel_matrix(X, kernel=self.kernel, gamma=self.gamma)

        self.coef_ = _kernel_logistic_regression_path(K, y, tol=self.tol, coef=None,
                                     C=self.C, solver='lbfgs', check_input=True,
                                     maxiter=1000)

        self.is_fitted_ = True

        return self

    def decision_function(self, X):

        check_is_fitted(self)

        if np.array_equal(self.X_.values, X.values):
            K = _kernel_matrix(X, kernel=self.kernel, gamma=self.gamma)
        else:
            K = _kernel_matrix(self.X_, Y=X, gamma=self.gamma, kernel=self.kernel).T
            
        scores = K.dot(self.coef_)

        return scores

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
        scores = self.decision_function(X)

        indices = (scores > 0).astype(np.int)

        return self.classes_[indices]
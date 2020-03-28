from KernelLogisticRegression import KernelLogisticRegression
import numpy as np
from scipy.special import expit
p = 2
n = 100
X = np.random.normal(0, 1, (n, p))
y = np.random.binomial(1, expit(X.sum(1)), n)
klr = KernelLogisticRegression(kernel="linear")
klr.fit(X, y)
klr.coef_
klr.decision_function(np.random.normal(0, 1, (n, p)))
klr.predict(X)
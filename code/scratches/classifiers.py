import sys
sys.path.extend(['/home/simon/Documents/601-Project/code'])
from data.pitchfx import PitchFxDataset
from models.classification.StrikeZoneLearner import StrikeZoneLearner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
# classifiers
from models.classification.kernel_logistic_regression import KernelLogisticRegression
from sklearn.svm import SVC
from pygam import LogisticGAM, te
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import pickle


pitchfx = PitchFxDataset()
df = pitchfx.group_by(
	umpire_HP="all",
	b_count=[0, 2, 3],
	s_count=[0, 1, 2]
)

szl = StrikeZoneLearner(scoring="neg_brier_score")

classifiers = []


# add SVC
svc = SVC(probability=True)
svc_params = {
	"C": np.logspace(-2, 1, 8),
	"gamma": np.logspace(-2, 1, 8),
	"class_weight": [None, "balanced"]
}
classifiers.append((svc, svc_params))

# add GAM
gam = LogisticGAM(te(0, 1))
gam_params = {
	"n_splines": [3, 5, 7, 10, 15],
	"lam": np.logspace(-2, 1, 10)
}
classifiers.append((gam, gam_params))

# add RandomForest
rf = RandomForestClassifier()
rf_params = {
	"n_estimators": np.logspace(1, 3, 10).round().astype(int),
	"ccp_alpha": np.logspace(-6, -1, 10),
	"class_weight": [None, "balanced"]
}
classifiers.append((rf, rf_params))

# add AdaBoost
ada = AdaBoostClassifier()
ada_params = {
	"n_estimators": np.logspace(1, 3, 10).round().astype(int),
	"learning_rate": np.logspace(-4, -1, 10)
}
classifiers.append((ada, ada_params))

# add MLP
mlp = MLPClassifier(max_iter=1000)
mlp_params = {
	"hidden_layer_sizes": [(8, ), (8, 8), (16, ), (16, 16), (8, 8, 8), (16, 16, 16)],
	"activation": ["relu", "logistic"],
	"alpha": np.logspace(-6, -1, 6)
}
classifiers.append((mlp, mlp_params))

# add Gradient Boosting
gdb = GradientBoostingClassifier()
gdb_params = {
	"n_estimators": np.logspace(1, 3, 10).round().astype(int),
	"learning_rate": np.logspace(-3, -1, 5)
}
classifiers.append((gdb, gdb_params))

# add KLR
klr = KernelLogisticRegression()
klr_params = {
	"gamma": np.logspace(-0.5, 0.5, 3),
	"C": np.logspace(-0.5, 0.5, 3)
}
classifiers.append((klr, klr_params))

# fit
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
	szl.fit_groups_all_classifiers(
		df=df,
		data_col=["px_std", "pz_std"],
		label_col="type_01",
		classifiers=classifiers,
		cv=True,
		cv_folds=5,
		n_jobs=-1,
	)




with open("./data/models/classifiers/umpire_balls_strikes_brier.txt", "wb") as f:
	pickle.dump(szl, f)


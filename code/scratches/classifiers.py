import sys
from data.pitchfx import PitchFxDataset
from models.classification.StrikeZoneLearner import StrikeZoneLearner
import numpy as np
import pickle

# classifiers
from models.classification.kernel_logistic_regression import KernelLogisticRegression
from sklearn.svm import SVC
from pygam import LogisticGAM, te
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from models.classification.polynomial_logistic_regression import PolynomialLogisticRegression

sys.path.extend(['/home/simon/Documents/601-Project/code'])
out_file = "./data/models/classifiers/umpire_pitchers_batters_auc_roc_svc_klr.txt"

pitchfx = PitchFxDataset()

df = pitchfx.group_by(
	umpire_HP="all",
	p_throws="all",
    stand="all"
)
szl = StrikeZoneLearner(scoring="roc_auc")

classifiers = []

# add SVC
svc = SVC(probability=True)
svc_params = {
	"C": np.logspace(-1, 1, 7),
	"gamma": np.logspace(-1, 0.3, 7),
	"class_weight": ["balanced"]
}
classifiers.append((svc, svc_params))

# # add RandomForest
# rf = RandomForestClassifier()
# rf_params = {
# 	"n_estimators": np.logspace(1, 3, 6).round().astype(int),
# 	"ccp_alpha": np.logspace(-6, -1, 6),
# 	"class_weight": [None, "balanced"]
# }
# classifiers.append((rf, rf_params))
#
# # add AdaBoost
# ada = AdaBoostClassifier()
# ada_params = {
# 	"n_estimators": np.logspace(1, 3, 6).round().astype(int),
# 	"learning_rate": np.logspace(-4, -1, 6)
# }
# classifiers.append((ada, ada_params))
#
# # add MLP
# mlp = MLPClassifier(max_iter=1000)
# mlp_params = {
# 	"hidden_layer_sizes": [(8, ), (8, 8), (16, ), (16, 16), (8, 8, 8), (16, 16, 16)],
# 	"activation": ["relu", "logistic"],
# 	"alpha": np.logspace(-6, -1, 6)
# }
# classifiers.append((mlp, mlp_params))
#
# # add Gradient Boosting
# gdb = GradientBoostingClassifier()
# gdb_params = {
# 	"n_estimators": np.logspace(1, 3, 10).round().astype(int),
# 	"learning_rate": np.logspace(-3, -1, 5)
# }
# classifiers.append((gdb, gdb_params))

# add KLR
klr = KernelLogisticRegression()
klr_params = {
	"gamma": np.logspace(-1, 1, 9),
	"C": np.logspace(-1, 1, 9)
}
classifiers.append((klr, klr_params))

# # add polynomial LogisticRegression
# plr = PolynomialLogisticRegression()
# plr_params = {
# 	"degree": [8, 10],
# 	"C": np.logspace(-2, 1, 4),
# 	"class_weight": [None, "balanced"]
# }
# classifiers.append((plr, plr_params))

# fit
szl.fit_groups_all_classifiers(
	df=df,
	data_col=["px_std", "pz_std"],
	label_col="type_01",
	classifiers=classifiers,
	cv=True,
	cv_folds=5,
	n_jobs=-1,
)

)
with open(out_file, "wb") as f:
	pickle.dump(szl, f)


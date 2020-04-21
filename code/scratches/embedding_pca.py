import sys
sys.path.extend(['/home/simon/Documents/601-Project/code'])
from data.pitchfx import PitchFxDataset
from models.classification.strikezone_learner import StrikezoneLearner
from sklearn.svm import SVC
from models.encoding.encoder import Encoder
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from plot.utils import plot_pitches


pitchfx = PitchFxDataset()
df = pitchfx.group_by(
    umpire_HP="all",
    b_count=[0, 2, 3],
    s_count=[0, 1, 2]
)

C = 0.7
gamma = 1.5

classifier = SVC(
    C=C,
    gamma=gamma,
    probability=True,
    class_weight="balanced"
)

szl = StrikezoneLearner(df, classifier, x_range=(-2, 2), y_range=(5, 0.5))

szl.predict_strikezone_all()

n_components = 7
encoder = PCA(n_components)
encoder = KernelPCA(
    n_components=n_components,
    kernel="rbf",
    gamma=0.01,
    alpha=0.00001,
    fit_inverse_transform=True
)


encoder = Encoder(encoder, "array")
encoder.fit(szl.strikezone)
# encoder.encoder.explained_variance_ratio_.cumsum()
embeddings = encoder.transform(szl.strikezone)
reconstructed_strikezones = encoder.inverse_transform(embeddings, szl.strikezone.keys())

lvls = ("Angel Hernandez", "b_count_[0,2]", "s_count_(1,2]")
pitches = df.get_group(lvls)


sz = reconstructed_strikezones[lvls].clip(0, 1)
plot_pitches(pitches=pitches, x_range=szl.x_range, z_range=szl.y_range, sz=sz)
plot_pitches(x_range=szl.x_range, z_range=szl.y_range, sz=sz)
plt.title(str(lvls) + " reconstructed with KPCA(" + str(n_components) + ")")
plt.show()


sz = szl.strikezone[lvls]
plot_pitches(pitches=pitches, x_range=szl.x_range, z_range=szl.y_range, sz=sz)
plot_pitches(x_range=szl.x_range, z_range=szl.y_range, sz=sz)
plt.title(str(lvls) + " original")
plt.show()
import sys
sys.path.extend(['/home/simon/Documents/601-Project/code'])
from data.pitchfx import PitchFxDataset
from models.classification.strikezone_learner import StrikezoneLearner
from sklearn.svm import SVC
from models.encoding.encoder import Encoder
from models.encoding.cnn_autoencoder import CNN, CNNAutoencoder
from models.encoding.encoder import strikezone_dict_to_tensor
from sklearn.decomposition import PCA
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

# n_components = 10
# # encoder = PCAStrikezone(n_components=n_components)
# encoder = Encoder(PCA(10), "array")
# encoder.fit(szl.strikezone)
# encoder.encoder.explained_variance_ratio_.cumsum()
# embeddings = encoder.transform(szl.strikezone)
# reconstructed_strikezones = encoder.inverse_transform(embeddings, szl.strikezone.keys())

cnn = CNN()
encoder = Encoder(CNNAutoencoder(cnn), "tensor")
encoder.fit(szl.strikezone)
embeddings = encoder.transform(szl.strikezone)
reconstructed_strikezones = encoder.inverse_transform(embeddings, szl.strikezone.keys())


# U = embeddings.cpu().numpy()[:, :, 0, 0]
#
# import torch
# import numpy as np
# self = encoder.encoder

# n_components = 7
# _, X = strikezone_dict_to_tensor(szl.strikezone)
# X = torch.tensor(np.expand_dims(X, axis=1)).float()
# # encoder
# X.size()
# X = torch.nn.Conv2d(1, 32, 8, stride=8, padding=4)(X)
# X.size()
# X = torch.nn.ReLU(True)(X)
# X.size()
# X = torch.nn.MaxPool2d(4, stride=4)(X)
# X.size()
# X = torch.nn.Conv2d(32, n_components, 3, stride=2, padding=1)(X)
# X.size()
# X = torch.nn.ReLU(True)(X)
# X.size()
# X = torch.nn.MaxPool2d(2, stride=1)(X)
# X.size()
# # decoder
# X = torch.nn.ConvTranspose2d(n_components, 32, 3, stride=1)(X)
# X.size()
# X = torch.nn.ReLU(True)(X)
# X.size()
# X = torch.nn.ConvTranspose2d(32, 8, 5, stride=2, padding=1)(X)
# X.size()
# X = torch.nn.ReLU(True)(X)
# X.size()
# X = torch.nn.ConvTranspose2d(8, 4, 3, stride=8, padding=1)(X)
# X.size()
# X = torch.nn.ReLU(True)(X)
# X.size()
# X = torch.nn.ConvTranspose2d(4, 1, 4, stride=2, padding=0)(X)
# X.size()
# X = torch.nn.Tanh()(X)
# X.size()


lvls = ("Angel Hernandez", "b_count_[0,2]", "s_count_(1,2]")
pitches = df.get_group(lvls)

sz = szl.strikezone[lvls]
# plot_pitches(pitches=pitches, x_range=szl.x_range, z_range=szl.y_range, sz=sz)
plot_pitches(x_range=szl.x_range, z_range=szl.y_range, sz=sz)
plt.title(str(lvls) + " original")
plt.show()

sz = reconstructed_strikezones[lvls].clip(0, 1)
# plot_pitches(pitches=pitches, x_range=szl.x_range, z_range=szl.y_range, sz=sz)
plot_pitches(x_range=szl.x_range, z_range=szl.y_range, sz=sz)
plt.title(str(lvls) + " reconstructed with PCA(" + str(n_components) + ")")
plt.show()
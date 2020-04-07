import sys
sys.path.extend(['/home/simon/Documents/601-Project/code'])
from data.pitchfx import PitchFxDataset
from models.classification.strikezone_learner import StrikezoneLearner
from sklearn.svm import SVC
from models.encoding.encoder import Encoder
from models.encoding.autoencoder import Autoencoder, CNNAutoencoder, NNAutoencoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from plot.utils import plot_pitches
import torch.nn
import pickle


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

encoder = NNAutoencoder(szl.res, n_components, (1024, 256, 64, 32))

# encoder = CNNAutoencoder(n_components=n_components)

encoder = Autoencoder(
    model=encoder,
    learning_rate=0.0001,
    num_epochs=50,
    weight_decay=0.000001,
    loss=torch.nn.functional.mse_loss
)
encoder = Encoder(encoder, "tensor")

encoder = PCA(n_components)

encoder = Encoder(encoder, "array")

encoder.fit(szl.strikezone)

embeddings = encoder.transform(szl.strikezone)
reconstructed_strikezones = encoder.inverse_transform(embeddings, szl.strikezone.keys())

with open("./data/models/encoding/PCA10_umpire_balls_strikes.txt", "wb") as f:
    pickle.dump((encoder, embeddings, szl.x_range, szl.y_range), f)

#
# import torch
# import numpy as np
# self = encoder.encoder
#
# U = embeddings.cpu().numpy()
#
# n_components = 7
#
#
# _,  X = strikezone_dict_to_tensor(szl.strikezone)
# X = torch.tensor(np.expand_dims(X, axis=1)).float()
# # encoder
# print(X.size())
# X = torch.nn.Conv2d(1, 4, 69, stride=1, padding=0)(X)
# print(X.size())
# X = torch.nn.Sigmoid()(X)
# print(X.size())
# X = torch.nn.Conv2d(4, 16, 25, stride=1, padding=0)(X)
# print(X.size())
# X = torch.nn.Sigmoid()(X)
# print(X.size())
# X = torch.nn.Conv2d(16, 128, 8, stride=1, padding=0)(X)
# print(X.size())
# X = torch.nn.Sigmoid()(X).reshape((-1, 128))
# print(X.size())
#
# X = torch.nn.Linear(128, 32)(X)
# print(X.size())
# X = torch.nn.ReLU(True)(X)
# print(X.size())
# X = torch.nn.Linear(32, n_components)(X)
# print(X.size())
# X = torch.nn.ReLU(True)(X)
# print(X.size())
#
# # decoder
# X = torch.nn.Linear(n_components, 32)(X)
# print(X.size())
# X = torch.nn.ReLU(True)(X)
# print(X.size())
# X = torch.nn.Linear(32, 128)(X)
# print(X.size())
# X = torch.nn.ReLU(True)(X)
# print(X.size())
# X = X.reshape((-1, 128, 1, 1))
# print(X.size())
# X = torch.nn.ConvTranspose2d(128, 16, 8, stride=1)(X)
# print(X.size())
# X = torch.nn.Sigmoid()(X)
# print(X.size())
# X = torch.nn.ConvTranspose2d(16, 4, 25, stride=1, padding=0)(X)
# print(X.size())
# X = torch.nn.Sigmoid()(X)
# print(X.size())
# X = torch.nn.ConvTranspose2d(4, 1, 69, stride=1, padding=0)(X)
# print(X.size())
# X = torch.nn.Sigmoid()(X)
# print(X.size())


lvls = ("Adam Hamari", "b_count_[0,2]", "s_count_(1,2]")
pitches = df.get_group(lvls)

for k, lvls in enumerate(szl.strikezone.keys()):
    if k > 20:
        break
    pitches = df.get_group(lvls)

    sz = szl.strikezone[lvls]
    plot_pitches(x_range=szl.x_range, z_range=szl.y_range, sz=sz,
                 sz_type="contour", X=szl.grid_x, Y=szl.grid_y)

    sz = reconstructed_strikezones[lvls].clip(0, 1)
    plot_pitches(pitches=None, x_range=szl.x_range,
                 z_range=szl.y_range, sz=sz,
                 sz_type="uncertainty", X=szl.grid_x, Y=szl.grid_y,
                 levels=[0.25, 0.5, 0.75])
    plt.title(str(lvls) + " reconstructed with PCA(" + str(n_components) + ")")
    plt.show()
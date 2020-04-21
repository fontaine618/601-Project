import numpy as np


def strikezone_dict_to_array(szd):
    levels = []
    strikezones = []
    for lvls, sz in szd.items():
        levels.append(lvls)
        strikezones.append(sz.ravel())
    strikezones = np.array(strikezones)
    return levels, strikezones


def strikezone_dict_to_tensor(szd):
    levels = list(szd.keys())
    strikezones = np.array(list(szd.values()))
    return levels, strikezones


def array_to_strikezone_dict(levels, arr):
    size = int(np.sqrt(arr.shape[1]))
    return {
        lvls: arr[i, :].reshape((size, size))
        for i, lvls in enumerate(levels)
    }


def tensor_to_strikezone_dict(levels, tensor):
    return {
        lvls: tensor[i, :, :]
        for i, lvls in enumerate(levels)
    }


class Encoder:

    def __init__(self, encoder, type="array"):
        self.encoder = encoder
        if type == "array":
            self.to_X = strikezone_dict_to_array
            self.to_dict = array_to_strikezone_dict
        elif type == "tensor":
            self.to_X = strikezone_dict_to_tensor
            self.to_dict = tensor_to_strikezone_dict

    def fit(self, d):
        _, X = self.to_X(d)
        self.encoder.fit(X)
        return self

    def transform(self, d):
        _, X = self.to_X(d)
        return self.encoder.transform(X)

    def inverse_transform(self, U, keys=None):
        X = self.encoder.inverse_transform(U)
        if keys is None:
            return X
        else:
            return self.to_dict(keys, X)

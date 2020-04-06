from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage


def generate_batter_outline():
    load = Image.open('./data/utils/batter_raw.png')
    gray = load.convert('L')
    gray = 255-np.array(gray)[::-1, :][:, ::-1]
    im = scipy.ndimage.zoom(gray, 3)
    contour = plt.contour(im, levels=[128])
    contour = contour.collections[0].get_paths()[0].vertices
    contour[:, 0] = contour[:, 0] * 2.5 / max(contour[:, 0])
    contour[:, 1] = contour[:, 1] - min(contour[:, 1])
    contour[:, 0] = contour[:, 0] - max(contour[:, 0])
    contour[:, 1] = contour[:, 1] * 6 / max(contour[:, 1])
    np.savetxt("./data/utils/batter_outline.csv", contour, delimiter=",")


def batter_outline(stand="R", x=-17/12):
    contour = np.loadtxt("./data/utils/batter_outline.csv", delimiter=",")
    contour[:, 0] += x
    if stand == "L":
        contour[:, 0] = -contour[:, 0]
    return contour[:, 0], contour[:, 1]


def strike_zone(x_range=(-10.5 / 12, 10.5 / 12), y_range=(1.603276, 3.425886)):
    # x_range: +/- 17/12 + 2
    # y_range: values used to standardize pz
    x = np.array([x_range[0], x_range[1], x_range[1], x_range[0], x_range[0]])
    y = np.array([y_range[0], y_range[0], y_range[1], y_range[1], y_range[0]])
    return x, y

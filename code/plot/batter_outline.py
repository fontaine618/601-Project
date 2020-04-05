from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage


def generate_contour():
    load = Image.open('./data/batter_outline/batter_raw.png')
    gray = load.convert('L')
    gray = 255-np.array(gray)[::-1, :][:, ::-1]
    im = scipy.ndimage.zoom(gray, 3)
    contour = plt.contour(im, levels=[128])
    contour = contour.collections[0].get_paths()[0].vertices
    contour[:, 0] = contour[:, 0] * 2.5 / max(contour[:, 0])
    contour[:, 1] = contour[:, 1] - min(contour[:, 1])
    contour[:, 0] = contour[:, 0] - max(contour[:, 0])
    contour[:, 1] = contour[:, 1] *7 / max(contour[:, 1])
    np.savetxt("./data/batter_outline/outline.csv", contour, delimiter=",")


def batter_outline(stand="R", x=-25.5/12):
    contour = np.loadtxt("./data/batter_outline/outline.csv", delimiter=",")
    contour[:, 0] += x
    if stand == "L":
        contour[:, 0] = -contour[:, 0]
    return contour[:, 0], contour[:, 1]
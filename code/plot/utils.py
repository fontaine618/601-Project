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
    contour = np.loadtxt("./../data/utils/batter_outline.csv", delimiter=",")
    contour[:, 0] += x
    if stand == "L":
        contour[:, 0] = -contour[:, 0]
    return contour[:, 0], contour[:, 1]


def strike_zone(x_range=(-10.5 / 12, 10.5 / 12), y_range=(1.603276, 3.425886)):
    # x_range: +/- 17/12*2 (plate half-width) + 2/12(ball diameter)
    # y_range: values used to standardize pz
    x = np.array([x_range[0], x_range[1], x_range[1], x_range[0], x_range[0]])
    y = np.array([y_range[0], y_range[0], y_range[1], y_range[1], y_range[0]])
    return x, y


def labeled_pitches(pitches):
    xb = pitches.loc[pitches["type"] == "B"]["px_std"]
    zb = pitches.loc[pitches["type"] == "B"]["pz_std"]
    xs = pitches.loc[pitches["type"] == "S"]["px_std"]
    zs = pitches.loc[pitches["type"] == "S"]["pz_std"]
    return xb, zb, xs, zs


def plot_pitches(pitches=None, x_range=(-2, 2), z_range=(0.5, 5),
                 sz=None, b_outline=True, sz_outline=True,
                 sz_type=None, levels=[0.5], X=None, Y=None):
    fig = plt.figure()
    if sz is not None:
        if sz_type == "heatmap":
            fig.imshow(sz, extent=(*x_range, *z_range[::-1]), alpha=sz.astype(float))
        elif sz_type == "contour":
            fig.contour(X, Y, sz, extent=(*x_range, *z_range[::-1]), levels=levels)
        elif sz_type == "contourf":
            fig.contourf(X, Y, sz, extent=(*x_range, *z_range[::-1]), levels=levels)
        elif sz_type == "uncertainty":
            fig.imshow(4 * sz * (1. - sz), extent=(*x_range, *z_range[::-1]),
                       alpha=(4 * sz * (1. - sz)).astype(float))
    else:
        fig.axis([*x_range, *z_range[::-1]])
    if b_outline:
        fig.plot(*batter_outline(), scalex=False, scaley=False, color="black")
    if sz_outline:
        fig.plot(*strike_zone(), scalex=False, scaley=False, color="white", linewidth=1, linestyle="--")
    if pitches is not None:
        xb, zb, xs, zs = labeled_pitches(pitches)
        fig.plot(xb, zb, label="Ball", scalex=False, scaley=False, linestyle="", marker="o")
        fig.plot(xs, zs, label="Strike", scalex=False, scaley=False, linestyle="", marker="o")
        fig.legend(framealpha=1., frameon=True, loc="upper right")
    return fig
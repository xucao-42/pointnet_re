import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import os
from pyntcloud import PyntCloud
import pandas as pd


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def display_point(pts, color, title=None, fname=None):
    """

    :param pts:
    :param color:
    :param title:
    :param fname:
    :return:
    """
    if isinstance(color, np.ndarray):
        color = np_color_to_hex_str(color)
    DPI =300
    PIX_h = 1000
    MARKER_SIZE = 5
    PIX_w = PIX_h

    X = pts[:, 0]
    Y = pts[:, 2]
    Z = pts[:, 1]
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    fig = plt.figure()
    fig.set_size_inches(PIX_w/DPI, PIX_h/DPI)

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=color, edgecolors="none", s=MARKER_SIZE, depthshade=True)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_title(title)
    ax.set_aspect("equal")
    plt.axis('off')
    if fname:
        plt.savefig(fname, transparent=True, dpi=DPI)
        plt.close(fig)
    else:
        plt.show()


def int16_to_hex_str(color):
    hex_str = ""
    color_map = {i: str(i) for i in range(10)}
    color_map.update({10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "F"})
    # print(color_map)
    hex_str += color_map[color // 16]
    hex_str += color_map[color % 16]
    return hex_str


def rgb_to_hex_str(*rgb):
    hex_str = "#"
    for item in rgb:
        hex_str += int16_to_hex_str(item)
    return hex_str


def np_color_to_hex_str(color):
    """
    :param color: an numpy array of shape (N, 3)
    :return: a list of hex color strings
    """
    hex_list = []
    for rgb in color:
        hex_list.append(rgb_to_hex_str(rgb[0], rgb[1], rgb[2]))
    return hex_list


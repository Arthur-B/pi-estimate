# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 00:02:43 2021

@author: Arthur
"""

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

sns.set()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# -----------------------------------------------------------------------------
# Prepare figure


def make_figure():
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Scatter plot
    circle = matplotlib.patches.Circle((0, 0), radius=1,
                                       alpha=0.1,
                                       edgecolor=colors[1],
                                       facecolor=colors[1])
    axs[0].add_patch(circle)
    axs[0].set(xlim=[-0.1, 1.1], ylim=[-0.1, 1.1],
               xticks=[0, 1], yticks=[0, 1],
               xlabel='x', ylabel='y', aspect="equal")

    # Error plot

    fig.tight_layout()
    return fig, axs


def compute_arc_length(nbPoints):

    error = np.zeros(len(nb_points))
    X = []
    Y = []

    for i in range(len(nb_points)):
        x = np.linspace(0, 1, nb_points[i])
        y = np.sqrt(1 - np.power(x, 2))

        x_edit = x[1:] - x[:-1]     # x2-x1
        y_edit = y[1:] - y[:-1]     # y2 -y1

        piApprox = 2 * np.sqrt(np.power(x_edit, 2) + np.power(y_edit, 2)).sum()
        error[i] = np.abs(piApprox - np.pi)

        X.append(x)
        Y.append(y)
    return X, Y, error


# -----------------------------------------------------------------------------
# Parameters


nb_points_start = np.linspace(np.log(2), np.log(100), 20)    # 20
nb_points_start = np.exp(nb_points_start)
nb_points_start = np.round(nb_points_start).astype(np.int)

nb_points_min = 100
nb_points_max = np.int(1e5)

nb_points = np.linspace(np.log(nb_points_min), np.log(nb_points_max), 30)  # 30
nb_points = np.exp(nb_points)
nb_points = np.round(nb_points).astype(np.int)

# add 2 to 9 (good points to understand what is going on)
nb_points = np.concatenate((nb_points_start[:-1], nb_points))


# Keep the points to make visualization
X, Y, error = compute_arc_length(nb_points)


# -----------------------------------------------------------------------------
# Figure: make and save


filenames = []

for i in range(len(nb_points)):

    # New figure
    fig, axs = make_figure()

    # Scatter plot until nb_point[i]
    axs[0].plot(X[i], Y[i], color=colors[0], alpha=0.5)

    # Plot the error
    axs[1].plot(nb_points[:i+1], error[:i+1], marker='o')

    axs[1].set(xlabel='Number of points (#)', ylabel='Absolute error',
               xscale='log', yscale='log',
               xlim=[1, 1e6], ylim=[1e-10, 1])

    # Save the frame and close the picture
    filename = f'temp/al_{i}.png'
    filenames.append(filename)
    fig.savefig(filename)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Build GIF

with imageio.get_writer('al.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

for filename in set(filenames):
    os.remove(filename)

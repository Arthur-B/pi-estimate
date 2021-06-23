# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:27:28 2020

@author: ArthurBaucour
"""

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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
    axs[0].set(xlim=[-1.1, 1.1], ylim=[-1.1, 1.1],
               xticks=[-1, 0, 1], yticks=[-1, 0, 1],
               xlabel='x', ylabel='y', aspect="equal")

    # Error plot

    fig.tight_layout()
    return fig, axs


def compute_mc(nb_points):

    error = np.zeros(len(nb_points))

    # Calculations for all points
    X = np.random.rand(nb_points_max, 2) * 2 - 1
    norm = np.linalg.norm(X, axis=1)
    con = np.where(norm < 1, True, False)

    for i in range(len(nb_points)):
        # Points in: X[con] or con.sum()   |     Points out: X[~con]
        pi_mc = 4 * con[:nb_points[i]].sum() / nb_points[i]
        # Get the error
        error[i] = np.abs(np.pi - pi_mc)

    return X[con], X[~con], error


# -----------------------------------------------------------------------------
# Parameters


nb_points_min = 10
nb_points_max = np.int(1e5)

nb_points = np.linspace(np.log(nb_points_min), np.log(nb_points_max), 50)
nb_points = np.exp(nb_points)
nb_points = np.round(nb_points).astype(np.int)


# -----------------------------------------------------------------------------
# Determine error for multiple tries (to get statistical)


nb_try = 20
error = np.zeros((nb_try, len(nb_points)))

for i in range(nb_try - 1):
    _, _, error[i] = compute_mc(nb_points)  # Only keep the error

# On last try, we keep the points to make visualization
X_in, X_out, error[-1] = compute_mc(nb_points)


# -----------------------------------------------------------------------------
# Build dataframe for error (with seaborn plot)


df = pd.DataFrame(data=error, columns=nb_points)
df = df.melt()
df = df.rename({'variable': 'Number of points (#)',
                'value': 'Absolute error'},
               axis='columns')


# -----------------------------------------------------------------------------
# Figure: make and save


filenames = []

for i in range(len(nb_points)):

    # New figure
    fig, axs = make_figure()

    # Scatter plot until nb_point[i]
    axs[0].scatter(X_in[:nb_points[i]][:, 0], X_in[:nb_points[i]][:, 1],
                   color=colors[0], marker='.', alpha=0.5)
    axs[0].scatter(X_out[:nb_points[i]][:, 0], X_out[:nb_points[i]][:, 1],
                   color=colors[1], marker='.', alpha=0.5)

    # Plot the error
    con = df['Number of points (#)'] <= nb_points[i]
    ax = sns.lineplot(x='Number of points (#)', y='Absolute error',
                      data=df[con], ax=axs[1], marker='o')

    axs[1].set(xlabel='Number of points (#)', ylabel='Absolute error',
               xscale='log', yscale='log',
               xlim=[1, 1e6], ylim=[1e-5, 1])

    # Save the frame and close the picture
    filename = f'temp/mc_{i}.png'
    filenames.append(filename)
    fig.savefig(filename)
    plt.close(fig)

# -----------------------------------------------------------------------------
# Build GIF

with imageio.get_writer('mc.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

for filename in set(filenames):
    os.remove(filename)

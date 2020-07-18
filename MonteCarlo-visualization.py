# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:27:28 2020

@author: ArthurBaucour
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def directNumpy(nbPoints):
    xy = np.random.rand(nbPoints,2) * 2 - 1
    norm = np.linalg.norm(xy, axis=1)
    nbIn = np.sum(norm < 1) # Count the booleans satisfying the conditions
    
    piApprox = 4 * nbIn / nbPoints
    error = np.abs(piApprox - np.pi)
    return piApprox, error

#------------------------------------------------------------------------------
# Visualization

# points

nbPointsVisualization = 2000
df_visualization = pd.DataFrame(data=np.random.rand(nbPointsVisualization,2) * 2 - 1,
                                columns=['x','y'])
df_visualization['norm'] = np.sqrt(df_visualization['x']**2 + df_visualization['y']**2)
df_visualization['Inside circle'] = df_visualization['norm'] < 1

# Figure

fig2, ax2 = plt.subplots()
circle = matplotlib.patches.Circle((0,0), radius=1,
                                    alpha=0.1,
                                    edgecolor=colors[1],
                                    facecolor=colors[1])
ax2.add_patch(circle)
ax2.set(aspect="equal", 
            xlim=[-1.1,1.1], ylim=[-1.1,1.1],
            xticks=[-1,0,1], yticks=[-1,0,1],
            xlabel='x', ylabel='y')

# title= 'Number of points: %i' % nbPointsVisualization

sns.scatterplot(x='x', y='y', data=df_visualization, hue='Inside circle',
                marker='.')
ax2.legend().remove()
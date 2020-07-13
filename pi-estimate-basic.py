# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 19:29:03 2020

@author: Arthur
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import time
import pandas as pd

def linear(nbPoints):
    countIn = 0

    for n in range(nbPoints):
        val = np.random.rand(2) * 2 - 1
        if np.linalg.norm(val) < 1:
            countIn +=1
    
    piApprox = 4 * countIn / nbPoints
    return piApprox


#==============================================================================
# Computing
#==============================================================================
    
#------------------------------------------------------------------------------
# Prepare dataframe for computing time and error

lsTry = [i for i in range(20)]
lsPointsTry = np.power(10, [i for i in range(3)])

index = pd.MultiIndex.from_product([lsPointsTry, lsTry],
                                   names=["Points", "Try"])

df = pd.DataFrame(data=np.zeros((len(index),2)),
                  index=index, columns=['Time (s)', 'Error'])


#------------------------------------------------------------------------------
# Computations

for ind in index:
    t = time.time()
    print("Nb of points: %i, Try: %i" % ind)
    nbPoints = ind[0]
    piApprox = linear(nbPoints)
    
    df['Error'].loc[ind] = np.abs(piApprox - np.pi)
    df['Time (s)'][ind]  = time.time() - t


#==============================================================================
# Figures
#==============================================================================

#------------------------------------------------------------------------------
# Time and error
    
fig1, axs1 = plt.subplots(1,2, figsize=(10,4))
axs1[0].set(xscale='log', yscale='log')
axs1[1].set(xscale='log', yscale='log')
sns.lineplot(x='Points', y='Time (s)', data=df.reset_index(), ax=axs1[0])
sns.lineplot(x='Points', y='Error', data=df.reset_index(), ax=axs1[1])

plt.tight_layout()


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

sns.scatterplot(x='x', y='y', data=df_visualization, hue='Inside circle')
ax2.legend().remove()


#------------------------------------------------------------------------------
# Show

plt.show()

    
    
    
    
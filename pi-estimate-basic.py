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

# Prepare series for computing time and error

lsTry = [i for i in range(20)]
lsPointsTry = np.power(10, [i for i in range(6)])

index = pd.MultiIndex.from_product([lsPointsTry, lsTry],
                                   names=["Points", "Try"])

df = pd.DataFrame(data=np.zeros((len(index),2)),
                  index=index, columns=['Time (s)', 'Error'])

# Prepare figure

# fig1, axs1 = plt.subplots(1,2)
# circle = matplotlib.patches.Circle((0,0), radius=0.5,
#                                    alpha=0.5,
#                                    edgecolor='k')
# axs1[0].add_patch(circle)
# axs1[0].set(aspect="equal", 
#             xlim=[-0.55,0.55], ylim=[-0.55,0.55],
#             xticks=[-0.5,0,0.5], yticks=[-0.5,0,0.5],
#             xlabel='x', ylabel='y')

# Linear, brute, understandable but maybe not efficient

for ind in index:
    t = time.time()
    print("Nb of points: %i, Try: %i" % ind)
    nbPoints = ind[0]
    piApprox = linear(nbPoints)
    
    df['Error'].loc[ind] = np.abs(piApprox - np.pi)
    df['Time (s)'][ind]  = time.time() - t
    
# Figure

fig1, axs1 = plt.subplots(1,2, figsize=(10,4))
axs1[0].set(xscale='log', yscale='log')
axs1[1].set(xscale='log', yscale='log')
sns.lineplot(x='Points', y='Time (s)', data=df.reset_index(), ax=axs1[0])
sns.lineplot(x='Points', y='Error', data=df.reset_index(), ax=axs1[1])

plt.tight_layout()
plt.show()

    
    
    
    
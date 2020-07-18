# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 19:29:03 2020

@author: Arthur

Brute gets linear from the start
Numpy gets linear from 1e5 points

Test: Numpy in batch, maybe processed in parallel
"""

import numpy as np
# import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import time
import pandas as pd


nbTry = 20  
maxPower = 6

# Total runtime
# 1e3: 1 s
# 1e4: 2.6 s
# 1e5: 17.7 s
# 1e6:

t_tot = time.time()

#==============================================================================
# Prepare dataframes for storage of time and error
#==============================================================================

lsTry = [i for i in range(nbTry)]
lsPointsTry = np.power(10, [i for i in range(maxPower + 1)])

index = pd.MultiIndex.from_product([lsPointsTry, lsTry],
                                   names=["Number of points", "Try"])

df_time = pd.DataFrame(data=np.zeros((len(lsPointsTry),2)),
                         index=lsPointsTry, columns=['Brute', 'Numpy'])
df_time.index.name = "Number of points"

df_error = pd.DataFrame(data=np.zeros((len(index),2)),
                         index=index, columns=['Brute', 'Numpy'])

#==============================================================================
# 1) Brute method: linear
#==============================================================================

#------------------------------------------------------------------------------
# Function

def linear(nbPoints):
    countIn = 0

    for n in range(nbPoints):
        val = np.random.rand(2) * 2 - 1
        if np.linalg.norm(val) < 1:
            countIn +=1
    
    piApprox = 4 * countIn / nbPoints
    error = np.abs(piApprox - np.pi)
    return piApprox, error

#------------------------------------------------------------------------------
# Computations
    
for nbPoints_ind in lsPointsTry:
    t = time.time()
    for nbTry_ind in lsTry:
        piApprox, error = linear(nbPoints_ind)
        df_error['Brute'].loc[nbPoints_ind, nbTry_ind] = error
    t = time.time() - t
    df_time['Brute'].loc[nbPoints_ind]  = t / nbTry
        
    
#==============================================================================
# 2) thinking a bit: Numpy vector operations
#==============================================================================
    
#------------------------------------------------------------------------------
# Function

def directNumpy(nbPoints):
    xy = np.random.rand(nbPoints,2) * 2 - 1
    norm = np.linalg.norm(xy, axis=1)
    nbIn = np.sum(norm < 1) # Count the booleans satisfying the conditions
    
    piApprox = 4 * nbIn / nbPoints
    error = np.abs(piApprox - np.pi)
    return piApprox, error

#------------------------------------------------------------------------------
# Calculations

for nbPoints_ind in lsPointsTry:
    t = time.time()
    for nbTry_ind in lsTry:
        piApprox, error = directNumpy(nbPoints_ind)
        df_error['Numpy'].loc[nbPoints_ind, nbTry_ind] = error
    t = time.time() - t
    df_time['Numpy'].loc[nbPoints_ind]  = t / nbTry


#==============================================================================
# Figures: time and error
#==============================================================================
  
fig1, axs1 = plt.subplots(1,2, figsize=(10,4))

sns.lineplot(data=df_time, ax=axs1[0], 
             dashes=False, markers=True)
axs1[0].set(xscale='log', yscale='log', ylabel='Time (s)')

sns.lineplot(data=df_error.droplevel('Try'), ax=axs1[1],
             dashes=False, markers=True)
axs1[1].set(xscale='log', yscale='log', ylabel='Error')

plt.tight_layout()

plt.show()


#==============================================================================
# Save
#==============================================================================

t_tot = time.time() - t_tot
print(t_tot)

# df_time.to_hdf('.\data\data_MonteCarlo.h5', key='df_time')
# df_error.to_hdf('.\data\data_MonteCarlo.h5', key='df_error',
#                 mode='w')
    
    
    
    
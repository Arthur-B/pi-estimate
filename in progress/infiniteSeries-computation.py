# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:21:43 2020
Viete --> Check geometric derivation
Wallis ok
Gregory-Leibniz (different values of z)

Fast ones:
Machin
Chudnovskys (14 digits per term)

Series acceleration?

@author: Arthur
"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
import time

#------------------------------------------------------------------------------
# Functions

def wallis(nbPoints):
    piApprox = 2
    for i in range(1,nbPoints+1):
        # print(i*2-1, i*2, i*2+1)
        piApprox *= (i*2)**2 / ((i*2-1) * (i*2+1))
    error = np.abs(piApprox - np.pi)
    return piApprox, error

def viete(nbPoints):
    a = np.sqrt(2)
    piApprox = 4 / a
    for i in range(1,nbPoints+1):
        a = np.sqrt(2 + a)
        piApprox *= np.divide(2, a)
    error = np.abs(piApprox - np.pi)
    return piApprox, error

#------------------------------------------------------------------------------
# Parameters
    
nbTry = 20  
maxPower = 5

t_tot = time.time()

#------------------------------------------------------------------------------
# Build Dataframe

# Multi Index

iterables = [['Viete', 'Wallis', 'Gregory-Leibniz'],
             ['Time', 'Error']]

multIndex = pd.MultiIndex.from_product(iterables, names=['Serie', 'Result'])

lsPointsTry = np.power(10, [i for i in range(maxPower + 1)])

df = pd.DataFrame(data=np.zeros((len(multIndex), len(lsPointsTry))),
                  columns=lsPointsTry, index=multIndex)

df.rename_axis("Number of Points", axis=1, inplace=True)

#------------------------------------------------------------------------------
# Calculations

# Wallis

for nbPointsTry in lsPointsTry:
    print('Wallis', nbPointsTry)
    t = time.time()
    for nbTry_ind in range(nbTry):
        piApprox, error = wallis(nbPointsTry)
        
    df[nbPointsTry].loc['Wallis','Error'] = error # Only store the last one
    t = time.time() - t
    df[nbPointsTry].loc['Wallis','Time'] = t / nbTry
    
# Viete

for nbPointsTry in lsPointsTry:
    print('Viete', nbPointsTry)
    t = time.time()
    for nbTry_ind in range(nbTry):
        piApprox, error = viete(nbPointsTry)
        
    df[nbPointsTry].loc['Viete','Error'] = error # Only store the last one
    t = time.time() - t
    df[nbPointsTry].loc['Viete','Time'] = t / nbTry
    

#==============================================================================
# Figures: time and error
#==============================================================================

# Process the df to fit in lineplot smoothly

df_graph = df.unstack(level=0).T.reset_index()

# Graphs

fig1, ax1 = plt.subplots()
sns.lineplot(x='Number of Points', y='Error', data=df_graph,
             hue='Serie', style='Serie', markers=True, dashes=False,
             ax=ax1)
ax1.set(xscale='log', yscale='log', 
        xlabel='Number of points', ylabel='Error')

plt.tight_layout()
# fig1, axs1 = plt.subplots(1,2, figsize=(10,4))

# sns.lineplot()

# # sns.lineplot(x='Number of Points', y='Error', data=df_graph, ax=axs1[0], 
# #               dashes=False, markers=True)
# # axs1[0].set(xscale='log', yscale='log', 
# #             xlabel='Number of points', ylabel='Time (s)')

# # sns.lineplot(x='Number of points', y='Error', data=df, ax=axs1[1],
# #               dashes=False, markers=True)
# # axs1[1].set(xscale='log', yscale='log', 
# #             xlabel='Number of points',ylabel='Error')

# plt.tight_layout()

# plt.show()


#==============================================================================
# Save
#==============================================================================

t_tot = time.time() - t_tot
print(t_tot)

# df.to_hdf('.\data\data_wallis.h5', key='df')


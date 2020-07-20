# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:35:42 2020

@author: ArthurBaucour
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Load the data

df_MonteCarlo =  pd.read_hdf('.\data\data_MonteCarlo.h5', key='df')
df_arcLength =  pd.read_hdf('.\data\data_arcLength.h5', key='df')

df = pd.concat({'Monte Carlo': df_MonteCarlo, 
                'Arc length': df_arcLength})

df = df.droplevel(1)
df.index.name = 'Method'
df = df.reset_index()
# df.set_index('Number of points', append=True)

#==============================================================================
# Figures: time and error
#==============================================================================
  
fig1, ax1 = plt.subplots()

sns.lineplot(x='Number of points', y ='Error',
             data=df, ax=ax1, 
             hue='Method', style='Method',
             dashes=False, markers=True)
ax1.set(xscale='log', yscale='log', ylabel='Error')

plt.tight_layout()

fig2, axs2 = plt.subplots(1,2, figsize=(8,4))

sns.lineplot(x='Number of points', y ='Time',
             data=df, ax=axs2[0],
             hue='Method', style='Method',
             dashes=False, markers=True)
axs2[0].set(xscale='log', yscale='log', ylabel='Time (s)')


sns.lineplot(x='Time', y ='Error',
             data=df, ax=axs2[1], 
             hue='Method', style='Method',
             dashes=False, markers=True)
axs2[1].set(xscale='log', yscale='log', ylabel='Error')

plt.tight_layout()

plt.show()
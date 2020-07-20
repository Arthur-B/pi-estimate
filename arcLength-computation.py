# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 19:09:35 2020

@author: ArthurBaucour

Define points following cosinus
"""
# import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
import pandas as pd
import time

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

nbTry = 20  
maxPower = 3

t_tot = time.time()

lsPointsTry = np.power(10, [i for i in range(maxPower + 1)])

df = pd.DataFrame(data=np.zeros((len(lsPointsTry),3)),
                  columns=['Number of points','Time', 'Error'])

df['Number of points'] = lsPointsTry

def arcLength(nbPoints):
    
    x = np.linspace(0,1,nbPoints)
    y = np.sqrt(1 - np.power(x,2))
    
    x_edit = x[1:] - x[:-1] # x2-x1
    y_edit = y[1:] - y[:-1] # y2 -y1
    
    piApprox = 2 * np.sqrt(np.power(x_edit,2) + np.power(y_edit, 2)).sum()
    error = np.abs(piApprox - np.pi)
    return piApprox, error


#------------------------------------------------------------------------------
# Calculations

for i in range(len(lsPointsTry)):
    t = time.time()
    for nbTry_ind in range(nbTry):
        piApprox, error = arcLength(lsPointsTry[i])
        
    df['Error'].iloc[i] = error # Only store the last one
    t = time.time() - t
    df['Time'].iloc[i] = t / nbTry


#==============================================================================
# Figures: time and error
#==============================================================================
  
fig1, axs1 = plt.subplots(1,2, figsize=(10,4))

sns.lineplot(x='Number of points', y='Time', data=df, ax=axs1[0], 
              dashes=False, markers=True)
axs1[0].set(xscale='log', yscale='log', 
            xlabel='Number of points', ylabel='Time (s)')

sns.lineplot(x='Number of points', y='Error', data=df, ax=axs1[1],
              dashes=False, markers=True)
axs1[1].set(xscale='log', yscale='log', 
            xlabel='Number of points',ylabel='Error')

plt.tight_layout()

plt.show()


#==============================================================================
# Save
#==============================================================================

t_tot = time.time() - t_tot
print(t_tot)

# df.to_hdf('.\data\data_arcLength.h5', key='df')
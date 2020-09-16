#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:07:58 2020

@author: jeff

An ever-expanding statistical analysis of team, game, and seasonal
features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statslib as st
import framelib as fl
import featurelib as feat
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
import sklearn.cluster as clustering
plt.close('all')

files = st.getFiles()
final_df = fl.arrangeFrame(files, scaling=PowerTransformer())
unscale_df = fl.arrangeFrame(files)[0]#.loc(axis=0)[:, 2015, :, :]
avdf = st.getSeasonalStats(unscale_df)
names = st.loadTeamNames(files)

sdf = final_df[0]#.loc(axis=0)[:, 2015, :, :]
sdf = sdf.drop(columns=['T_Elo', 'O_Elo', 'T_Rank', 'O_Rank']) #Remove overall measures of team
ssdf = sdf.drop(columns=['T_Score', 'O_Score'])
tsdf = sdf[[s for s in sdf.columns if s[:2] == 'T_']]

#%%
'''
OFFENSIVE INDICATORS
'''

#Initial testing: we have our significant variables, let's try to reduce them to something useful.
eigdf = tsdf.drop(columns=['T_Score'])
U, sig, Vt = np.linalg.svd(eigdf, full_matrices=False)
plt.figure('Eigs')
plt.subplot(2, 1, 1)
plt.plot(sig)
plt.subplot(2, 1, 2)
plt.plot(np.gradient(sig))

ocorrs = np.corrcoef(U.T, unscale_df['T_Score'])[-1, :-1]
dcorrs = np.corrcoef(U.T, unscale_df['O_Score'])[-1, :-1]

plt.figure('Eig Corrs')
plt.plot(abs(ocorrs))
plt.plot(abs(dcorrs))

# plt.figure()
# plt.plot(U[0, :])
# plt.plot(eigdf.iloc[0].dot(np.linalg.pinv(Vt)).dot(np.linalg.pinv(np.diag(sig))))

oedf = pd.DataFrame(index=sdf.index, columns=np.arange(len(ocorrs))[abs(ocorrs) > .1], data=U[:, abs(ocorrs) > .1])
dedf = pd.DataFrame(index=sdf.index, columns=np.arange(len(dcorrs))[abs(dcorrs) > .1], data=U[:, abs(dcorrs) > .1])

plt.figure('Histograms')
for c in oedf.columns:
    hist, bins = np.histogram(oedf[c], bins='fd')
    plt.plot(bins[1:], hist)
    
#As gaussian random variables, we can get a mean and variance and find the outliers in every feature
omeans = oedf.groupby(['Season', 'TID']).mean()
omeans = (omeans - omeans.mean()) / omeans.std()

pltavdf = (avdf - avdf.mean()) / avdf.std()
#Let's only keep the variables we think might show differences in offensive approach
pltavdf = pltavdf[['T_Score', 'T_Ast', 'T_eFG%', 'T_Econ', 'T_Poss', 'T_FT/A', 'T_Ast%', 'T_3Two%', 'O_Elo', 'O_Rank']]
plt.figure('Off. Measurables')
grid_sz = np.ceil(np.sqrt(omeans.shape[1])).astype(int)
for n, c in enumerate(omeans.columns):
    pltavdf['Category'] = 0
    pltavdf.loc[omeans[c] >= 1, 'Category'] = 1
    pltavdf.loc[omeans[c] <= -1, 'Category'] = -1
    plt.subplot(grid_sz, grid_sz, n+1)
    sns.violinplot(x='variable', y='value', hue='Category', data=pltavdf.melt(id_vars=['Category']))
    plt.xticks(rotation=45)
    plt.title('O{}'.format(c))
    
dmeans = dedf.groupby(['Season', 'TID']).mean()
dmeans = (dmeans - dmeans.mean()) / dmeans.std()
pltavdf = (avdf - avdf.mean()) / avdf.std()
#Let's only keep the variables we think might show differences in defensive approach
pltavdf = pltavdf[['O_Score', 'O_Ast', 'T_R%', 'O_PF', 'O_ProdPoss%', 'O_eFG%', 'O_Elo', 'O_Rank']]
plt.figure('Def. Measurables')
grid_sz = np.ceil(np.sqrt(dmeans.shape[1])).astype(int)
for n, c in enumerate(dmeans.columns):
    pltavdf['Category'] = 0
    pltavdf.loc[dmeans[c] >= 1, 'Category'] = 1
    pltavdf.loc[dmeans[c] <= -1, 'Category'] = -1
    plt.subplot(grid_sz, grid_sz, n+1)
    sns.violinplot(x='variable', y='value', hue='Category', data=pltavdf.melt(id_vars=['Category']))
    plt.xticks(rotation=45)
    plt.title('D{}'.format(c))
    
#Let's look at some specific stylistic differences between teams
#OFFENSIVE STYLES
plt.figure('Offensive Style')
bsc = omeans.loc[[i in [names['Gonzaga'], names['St Mary\'s CA'], names['N Kentucky']] for i in omeans.index.get_level_values(1)]].loc(axis=0)[2019, :]
fast = omeans.loc[[i in [names['North Carolina'], names['Buffalo'], names['Belmont'], names['Duke']] for i in omeans.index.get_level_values(1)]].loc(axis=0)[2019, :]
slow = omeans.loc[[i in [names['Virginia'], names['St Mary\'s CA'], names['Virginia Tech'], names['Liberty']] for i in omeans.index.get_level_values(1)]].loc(axis=0)[2019, :]
sns.scatterplot(x='variable', y='value', data=fast.melt())
sns.scatterplot(x='variable', y='value', data=slow.melt())

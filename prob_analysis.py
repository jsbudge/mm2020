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


def runProbs(p, gr, val):
    ncats = len(list(set(gr)))
    ret = np.zeros((ncats, p.shape[1]))
    for n in range(ncats):
        mu = p.loc[gr == n].mean().values
        std = p.loc[gr == n].std().values
        ret[n, :] = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-.5 * ((val - mu) / std)**2)
    for n in range(p.shape[1]):
        ret[:, n] = ret[:, n] / np.sum(ret[:, n])
    return ret

plt.close('all')

files = st.getFiles()
final_df = fl.arrangeFrame(files, scaling=StandardScaler())
unscale_df = fl.arrangeFrame(files)[0]#.loc(axis=0)[:, 2015, :, :]
tdf, twin, _ = fl.arrangeTourneyGames(files)
ttdf = st.getTourneyStats(tdf, unscale_df, files)
avdf = st.getSeasonalStats(unscale_df, strat='gausselo')
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
eigdf = tsdf.drop(columns=['T_Score', 'T_ScoreShift'])
U, sig, Vt = np.linalg.svd(eigdf, full_matrices=False)
plt.figure('Eigs')
plt.subplot(2, 1, 1)
plt.plot(sig)
plt.subplot(2, 1, 2)
plt.plot(np.gradient(sig))

ocorrs = np.corrcoef(U.T, unscale_df['T_Score'])[-1, :-1]
dcorrs = np.corrcoef(U.T, unscale_df['T_ScoreShift'])[-1, :-1]

plt.figure('Eig Corrs')
plt.plot(abs(ocorrs))
plt.plot(abs(dcorrs))

# plt.figure()
# plt.plot(U[0, :])
# plt.plot(eigdf.iloc[0].dot(np.linalg.pinv(Vt)).dot(np.linalg.pinv(np.diag(sig))))
oeigs = np.logical_and(abs(ocorrs) > abs(dcorrs), abs(ocorrs) > .1)
deigs = np.logical_and(abs(dcorrs) > abs(ocorrs), abs(dcorrs) > .1)
oedf = pd.DataFrame(index=sdf.index, columns=np.arange(len(ocorrs))[oeigs], data=U[:, oeigs])
oedf = oedf.rename(columns=lambda x: 'O{}'.format(x))
oedf = st.normalizeToSeason(oedf, scaler=StandardScaler()) * 16.666 + 50
dedf = pd.DataFrame(index=sdf.index, columns=np.arange(len(dcorrs))[deigs], data=U[:, deigs])
dedf = dedf.rename(columns=lambda x: 'D{}'.format(x))
dedf = st.normalizeToSeason(dedf, scaler=StandardScaler()) * 16.666 + 50

plt.figure('Histograms')
for c in oedf.columns:
    hist, bins = np.histogram(oedf[c], bins='fd')
    plt.plot(bins[1:], hist)
    
#As gaussian random variables, we can get a mean and variance and find the outliers in every feature
omeans = oedf.groupby(['Season', 'TID']).mean()
omeans['Ovar'] = np.sqrt(np.sum(oedf.groupby(['Season', 'TID']).std()**2, axis=1))

dmeans = dedf.groupby(['Season', 'TID']).mean()
dmeans['Dvar'] = np.sqrt(np.sum(dedf.groupby(['Season', 'TID']).std()**2, axis=1))

fulltdf = pd.DataFrame(index=tdf.index).join(omeans, on=['Season', 'TID']).join(dmeans, on=['Season', 'TID'])
fullodf = fulltdf - pd.DataFrame(index=tdf.index).join(omeans, on=['Season', 'OID']).join(dmeans, on=['Season', 'OID'])

pltavdf = (avdf - avdf.mean()) / avdf.std()
#Let's only keep the variables we think might show differences in offensive approach
pltavdf = pltavdf[['T_Score', 'T_Ast', 'T_eFG%', 'T_Econ', 'T_Poss', 'T_FT/A', 'T_Ast%', 'T_3Two%', 'O_Elo', 'O_Rank']]
plt.figure('Off. Measurables')
grid_sz = np.ceil(np.sqrt(omeans.shape[1])).astype(int)
for n, c in enumerate(omeans.columns):
    pltavdf['Category'] = 0
    pltavdf.loc[omeans[c] >= 67, 'Category'] = 1
    pltavdf.loc[omeans[c] <= 33, 'Category'] = -1
    plt.subplot(grid_sz, grid_sz, n+1)
    sns.violinplot(x='variable', y='value', hue='Category', data=pltavdf.melt(id_vars=['Category']))
    plt.xticks(rotation=45)
    plt.title(c)
    
pltavdf = (avdf - avdf.mean()) / avdf.std()
#Let's only keep the variables we think might show differences in defensive approach
pltavdf = pltavdf[['O_Score', 'O_Ast', 'T_R%', 'O_PF', 'O_ProdPoss%', 'O_eFG%', 'O_Elo', 'O_Rank']]
plt.figure('Def. Measurables')
grid_sz = np.ceil(np.sqrt(dmeans.shape[1])).astype(int)
for n, c in enumerate(dmeans.columns):
    pltavdf['Category'] = 0
    pltavdf.loc[dmeans[c] >= 67, 'Category'] = 1
    pltavdf.loc[dmeans[c] <= 33, 'Category'] = -1
    plt.subplot(grid_sz, grid_sz, n+1)
    sns.violinplot(x='variable', y='value', hue='Category', data=pltavdf.melt(id_vars=['Category']))
    plt.xticks(rotation=45)
    plt.title(c)

#Looking at some stacked plots of probabilities, given the value of each feature
stackdf = fullodf.merge(ttdf[['GameRank', 'T_Seed', 'T_FinalRank']], on=['Season', 'TID'])
gr = stackdf['GameRank']
stackdf = stackdf.drop(columns=['GameRank'])
grid_sz = np.ceil(np.sqrt(stackdf.shape[1])).astype(int)
pts = np.linspace(0, 100, 200)
stack = np.zeros((7, stackdf.shape[1], len(pts)))
for n in range(len(pts)):
    stack[:, :, n] = runProbs(stackdf, gr, pts[n])
plt.figure('Tourn. Dist')
for n in range(stackdf.shape[1]):
    plt.subplot(grid_sz, grid_sz, n+1)
    plt.stackplot(pts, stack[:, n, :])
    plt.title(stackdf.columns[n])
plt.legend(['R128', 'R64', 'R32', 'S16', 'E8', 'F4', 'CH'], loc='lower right')

#Check how each feature impacts the chance of winning
stackdf = oedf.join(dedf)
gr = (unscale_df['T_Score'] - unscale_df['O_Score'] > 0).astype(int)
grid_sz = np.ceil(np.sqrt(stackdf.shape[1])).astype(int)
pts = np.linspace(0, 100, 200)
stack = np.zeros((2, stackdf.shape[1], len(pts)))
for n in range(len(pts)):
    stack[:, :, n] = runProbs(stackdf, gr, pts[n])
plt.figure('WinProb')
for n in range(stackdf.shape[1]):
    plt.subplot(grid_sz, grid_sz, n+1)
    plt.stackplot(pts, stack[:, n, :])
    plt.title(stackdf.columns[n])
plt.legend(['Loss', 'Win'], loc='lower right')

# for yr in range(2003, 2020):
#     try:
#         byuprobs = np.sum(runProbs(stackdf, gr, stackdf.loc(axis=0)[yr, names['St Mary\'s CA']].values[0, :]), axis=1)
#         byuprobs = byuprobs / sum(byuprobs)
#         print('{}: {} GP, {:.2f}%'.format(yr, np.arange(7)[byuprobs == byuprobs.max()][0], byuprobs[byuprobs == byuprobs.max()][0] * 100))
#     except:
#         print('Missed {}'.format(yr))
    
#Let's look at some specific stylistic differences between teams
#OFFENSIVE STYLES
test = omeans['O0'] > 60
plt.figure('ShiftDist')
sns.kdeplot(avdf.loc[test, 'T_Score'])
sns.kdeplot(avdf.loc[np.logical_not(test), 'T_Score'])



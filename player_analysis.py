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
from tqdm import tqdm
from itertools import combinations
import framelib as fl
import featurelib as feat
import seaborn as sns
import eventlib as ev
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures, OneHotEncoder
from sklearn.decomposition import KernelPCA
import sklearn.cluster as cl
from scipy.stats import multivariate_normal as mvn
from scipy.stats import iqr
from collections import Counter
plt.close('all')

files = st.getFiles()
def findPlayerTeam(games, rosters, pid):
    r = rosters.loc(axis=0)[:, pid]
    return Counter(list(games.loc[r.index.get_level_values(0)].values.flatten())).most_common(1)[0][0]
        
season = 2017
# av_df = pd.DataFrame()
# adv_df = pd.DataFrame()
# for seas in [2015, 2016, 2017, 2018, 2019, 2020]:
#     games, rosters = ev.getRosters(files, seas)
#     sum_df = rosters.groupby(['PlayerID']).sum()
#     tmp_df = ev.getAdvStats(sum_df)
#     tmp_df['Season'] = seas
#     tmp_df['Mins'] = sum_df['Mins']
#     tmp_df = tmp_df.reset_index().set_index(['Season', 'PlayerID'])
#     av_df = av_df.append(tmp_df)
#     tmp_df = rosters.groupby(['PlayerID']).mean()
#     tmp_df['Season'] = seas
#     tmp_df = tmp_df.reset_index().set_index(['Season', 'PlayerID'])
#     adv_df = adv_df.append(tmp_df)
# av_df.to_csv('./data/AVRosterData.csv')
# adv_df.to_csv('./data/MeanRosterData.csv')
av_df = pd.read_csv('./data/AVRosterData.csv').set_index(['Season', 'PlayerID'])
adv_df = pd.read_csv('./data/MeanRosterData.csv').set_index(['Season', 'PlayerID'])
#%%
games, rosters = ev.getRosters(files, season)
av_df = av_df.loc[av_df['Mins'] > 36]
ov_perc = (av_df - av_df.mean()) / av_df.std()

# sdf = fl.arrangeFrame(files, scaling=None, noinfluence=True)[0].loc(axis=0)[:, season, :, :]
# savdf = st.getSeasonalStats(sdf, strat='relelo')
#%%
    
shifter = np.ones((ov_perc.shape[1],))
shifter[2:4] = -1
off_cons = np.array([1, 1, 0, 0, 1, 0, .7, 0, 0, .5, .5, .33, .5, .33, .33, 1, 1, 1, 1, 0])
def_cons = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
score_cons = off_cons + def_cons
adj_mins = av_df['Mins']
adj_mins = adj_mins.loc[ov_perc.index]
av_df['2WayScore'] = np.sum(ov_perc * shifter * score_cons, axis=1)
av_df['OffScore'] = np.sum(ov_perc * shifter * off_cons, axis=1)
av_df['DefScore'] = np.sum(ov_perc * shifter * def_cons, axis=1)
av_df['SpecScore'] = np.max(ov_perc * shifter, axis=1)
av_df['ShiftScore'] = (np.max(ov_perc * shifter, axis=1) + np.min(ov_perc * shifter, axis=1))
#av_df['Mins'] = adj_mins * 40

#CLUSTERING TO FIND PLAYER TYPES
n_types = 5

#Get number of clusters that best distinguishes between players
cluster_algs = [
                    cl.KMeans(n_clusters=n_types)
                ]

kpca = KernelPCA(n_components=5)
pca_df = pd.DataFrame(index=av_df.index, data=kpca.fit_transform(av_df))
cat_perc = pca_df.copy()
for idx, c in enumerate(cluster_algs):
    cat = c.fit_predict(pca_df)
for n in range(len(list(set(cluster_algs[-1].labels_)))):
    cptmp = pca_df.loc[cat == n]
    cat_perc.loc[cat == n] = (cptmp - cptmp.mean()) / cptmp.std()
cat_nme = 'C{}'.format(idx)
av_df[cat_nme] = cat
cat_df = av_df.groupby([cat_nme]).mean()
av_df['CatScore'] = np.sum(cat_perc, axis=1)
for col in av_df.columns:
    if 'Score' in col:
        av_df[col] = ((av_df[col] - av_df[col].mean()) / av_df[col].std()) * 15 + 50
    

#%%

wtids = list(set(games[['WTeamID', 'LTeamID']].values.flatten()))
ts_cols = []
for col in av_df.columns:
    if 'Score' in col:
        ts_cols = ts_cols + [col, col[:-5] + 'WScore', col[:-5] + 'Spread']
ts_df = pd.DataFrame(index=wtids, columns=ts_cols)
for i in wtids:
    gids = games.loc[np.logical_or(games['WTeamID'] == i, games['LTeamID'] == i)]
    rids = rosters.loc[gids.index]
    tap = Counter(rids.index.get_level_values(1))
    team_players = [p[0] for p in tap.most_common(15) if p[0] in av_df.loc(axis=0)[season, :].index.get_level_values(1) and p[1] >= 3]
    wghts = adj_mins.loc(axis=0)[season, team_players].values
    tmp_df = av_df.loc(axis=0)[season, team_players]
    if len(wghts) > 0:
        ts_vals = []
        for col in av_df.columns:
            if 'Score' in col:
                ts_vals = ts_vals + [np.mean(tmp_df[col]),
                                     np.average(tmp_df[col].values, weights=wghts),
                                     np.std(tmp_df[col])]
        ts_df.loc[i, ts_cols] = ts_vals
ts_df = ts_df.astype(float)

#%%

#Lineup experimentation
g_ev, line, sec_df, l_df = ev.getSingleGameLineups(4717, files, season)

lin_eval = pd.DataFrame(columns=av_df.columns)
for l in line[0]:
    try:
        lin_eval.loc[l] = av_df.loc(axis=0)[season, line[0][l]].mean()
    except:
        print('Lineup contains non-impact player.')



#%%
plt.close('all')
#Figures
plt_cols = ['AstPer18', 'FoulPer18', 'BlkPer18', 'TOPer18', 
            'PtsPer18', 'eFG%', 'FT/A', cat_nme]
sns.pairplot(av_df.loc[av_df[cat_nme] != -1, plt_cols], hue=cat_nme, plot_kws={'s': 15})
    







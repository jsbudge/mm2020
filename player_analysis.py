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
import sklearn.cluster as cl
from scipy.stats import multivariate_normal as mvn
from scipy.stats import iqr
from collections import Counter
plt.close('all')

files = st.getFiles()
def findPlayerTeam(games, rosters, pid):
    r = rosters.loc(axis=0)[:, pid]
    return Counter(list(games.loc[r.index.get_level_values(0)].values.flatten())).most_common(1)[0][0]
        

games, rosters = ev.getRosters(files, 2017)
#%%

p_df = rosters.groupby(['PlayerID']).sum()
av_df = ev.getAdvStats(p_df)
av_df = av_df.loc[rosters.groupby(['PlayerID']).sum()['Mins'] > 36]
ov_perc = (av_df - av_df.mean()) / av_df.std()
cat_perc = ov_perc.copy()
adv_df = rosters.groupby(['PlayerID']).mean()
p_df = (p_df - p_df.mean()) / p_df.std()
sdf = fl.arrangeFrame(files, scaling=None, noinfluence=True)[0].loc(axis=0)[:, 2017, :, :]
savdf = st.getSeasonalStats(sdf, strat='relelo')
#%%
#CLUSTERING TO FIND PLAYER TYPES
n_types = 5

#Get number of clusters that best distinguishes between players
cluster_algs = [
                    cl.KMeans(n_clusters=n_types)
                ]
for idx, c in enumerate(cluster_algs):
    cat = c.fit_predict(av_df)
    for n in range(n_types):
        cptmp = av_df.loc[cat == n]
        cat_perc.loc[cat == n] = (cptmp - cptmp.mean()) / cptmp.std()
    av_df['C{}'.format(idx)] = cat
    
shifter = np.ones((ov_perc.shape[1],))
shifter[2:4] = -1
score_cons = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, .5, .5, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1])
adj_mins = rosters.groupby(['PlayerID']).sum()['Mins'] / 1320
adj_mins = adj_mins.loc[ov_perc.index]
av_df['OvScore'] = np.sum(ov_perc * shifter * score_cons, axis=1) * adj_mins
av_df['CatScore'] = np.sum(cat_perc * shifter * score_cons, axis=1) * adj_mins
av_df['SpecScore'] = np.max(ov_perc * shifter, axis=1) * adj_mins
av_df['ShiftScore'] = (np.max(ov_perc * shifter, axis=1) + np.min(ov_perc * shifter, axis=1)) * adj_mins
for col in av_df.columns:
    if 'Score' in col:
        av_df[col] = ((av_df[col] - av_df[col].mean()) / av_df[col].std()) * 15 + 50
    

#%%

wtids = list(set(games[['WTeamID', 'LTeamID']].values.flatten()))
ts_df = pd.DataFrame(index=wtids, columns=['OvScore', 'OvWScore', 'OvSpread', 
                  'CatScore', 'CatWScore', 'CatSpread',
                  'SpecScore', 'SpecWScore', 'SpecSpread',
                  'ShiftScore', 'ShiftWScore', 'ShiftSpread'])
for i in wtids:
    gids = games.loc[np.logical_or(games['WTeamID'] == i, games['LTeamID'] == i)]
    rids = rosters.loc[gids.index]
    tap = Counter(rids.index.get_level_values(1))
    team_players = [p[0] for p in tap.most_common(15) if p[0] in av_df.index and p[1] >= 3]
    wghts = adj_mins.loc[team_players].values
    ts_df.loc[i, ['OvScore', 'OvWScore', 'OvSpread', 
                  'CatScore', 'CatWScore', 'CatSpread',
                  'SpecScore', 'SpecWScore', 'SpecSpread',
                  'ShiftScore', 'ShiftWScore', 'ShiftSpread']] = [np.average(av_df.loc[team_players, 'OvScore']),
                    np.average(av_df.loc[team_players, 'OvScore'], weights=wghts),
                    np.std(av_df.loc[team_players, 'OvScore']),
                    np.average(av_df.loc[team_players, 'CatScore']),
                    np.average(av_df.loc[team_players, 'CatScore'], weights=wghts),
                    np.std(av_df.loc[team_players, 'CatScore']),
                    np.average(av_df.loc[team_players, 'SpecScore']),
                    np.average(av_df.loc[team_players, 'SpecScore'], weights=wghts),
                    np.std(av_df.loc[team_players, 'SpecScore']),
                    np.average(av_df.loc[team_players, 'ShiftScore']),
                    np.average(av_df.loc[team_players, 'ShiftScore'], weights=wghts),
                    np.std(av_df.loc[team_players, 'ShiftScore'])]
ts_df = ts_df.astype(float)





    







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
#%%

games, rosters = ev.getRosters(files, 2017)
p_df = rosters.groupby(['PlayerID']).sum()
av_df = ev.getAdvStats(p_df)
av_df = av_df.loc[rosters.groupby(['PlayerID']).mean()['Mins'] > 6]
ov_perc = (av_df - av_df.mean()) / av_df.std()
cat_perc = ov_perc.copy()
#av_df['Mins'] = rosters['Mins']
#av_df = av_df.groupby(['PlayerID']).mean()
p_df = (p_df - p_df.mean()) / p_df.std()

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
score_cons = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, .33, .33, .33, 1, 1, 1, 1])
    
av_df['OvScore'] = np.sum(ov_perc * shifter * score_cons, axis=1)
av_df['CatScore'] = np.sum(cat_perc * shifter * score_cons, axis=1)
    

#%%

wtids = list(set(games['WTeamID']))
ts_df = pd.DataFrame(index=wtids, columns=['OvScore', 'OvWScore', 'OvSpread', 
                                           'CatScore', 'CatWScore', 'CatSpread'])
for i in wtids:
    gids = games.loc[np.logical_or(games['WTeamID'] == i, games['LTeamID'] == i)]
    rids = rosters.loc[gids.index]
    tap = Counter(rids.index.get_level_values(1))
    team_players = [p[0] for p in tap.most_common(15) if p[0] in av_df.index]
    ts_df.loc[i, ['OvScore', 'OvWScore', 'OvSpread', 
                  'CatScore', 'CatWScore', 'CatSpread']] = [np.average(av_df.loc[team_players, 'OvScore']),
                    np.average(av_df.loc[team_players, 'OvScore'], weights=p_df.loc[team_players, 'Mins']),
                    np.std(av_df.loc[team_players, 'OvScore']),
                    np.average(av_df.loc[team_players, 'CatScore']),
                    np.average(av_df.loc[team_players, 'CatScore'], weights=p_df.loc[team_players, 'Mins']),
                    np.std(av_df.loc[team_players, 'CatScore'])]





    







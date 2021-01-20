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

def getPlayerData(team, rosters, games, df, season=2017):
    gids = games.loc[np.logical_or(games['WTeamID'] == team, 
                                   games['LTeamID'] == team)]
    rids = rosters.loc[gids.index]
    tap = Counter(rids.index.get_level_values(1))
    return [p[0] for p in tap.most_common(15) if p[0] in df.loc(axis=0)[season, :].index.get_level_values(1) and p[1] >= 3]

def getTeamRosterData(team, rosters, games, df, season):
    return df.loc(axis=0)[season, getPlayerData(team, rosters, games, df, season)]
        
season = 2019
# av_df = pd.DataFrame()
# adv_df = pd.DataFrame()
# for seas in [2015, 2016, 2017, 2018, 2019, 2020]:
    # sdf = fl.arrangeFrame(files, season, scaling=None, noinfluence=True)[0]
    # games, rosters = ev.getRosters(files, seas)
    # sum_df = rosters.groupby(['PlayerID']).sum()
    # tmp_df = ev.getAdvStats(sum_df)
    # tmp_df['Season'] = seas
    # tmp_df['Mins'] = sum_df['Mins']
    # tmp_df = tmp_df.reset_index().set_index(['Season', 'PlayerID'])
    # av_df = av_df.append(tmp_df)
    # tmp_df = rosters.groupby(['PlayerID']).mean()
    # tmp_df['Season'] = seas
    # tmp_df = tmp_df.reset_index().set_index(['Season', 'PlayerID'])
    # adv_df = adv_df.append(tmp_df)
# av_df.to_csv('./data/AVRosterData.csv')
# adv_df.to_csv('./data/MeanRosterData.csv')
print('Loading player data...')
av_df = pd.read_csv('./data/AVRosterData.csv').set_index(['Season', 'PlayerID'])
m_df = pd.read_csv('./data/MeanRosterData.csv').set_index(['Season', 'PlayerID'])
pdf = ev.getTeamRosters(files)
av_df = av_df.merge(pdf['TeamID'], on='PlayerID', right_index=True)
av_df = av_df.reset_index().set_index(['Season', 'PlayerID', 'TeamID'])
m_df = m_df.merge(pdf['TeamID'], on='PlayerID', right_index=True)
m_df = m_df.reset_index().set_index(['Season', 'PlayerID', 'TeamID'])
games, roster_df = ev.getRosters(files, season)
sdf = fl.arrangeFrame(files, season, scaling=None, noinfluence=True)[0]
sdf1 = sdf[['DayNum', 'T_Poss', 'T_Ast', 'T_Score', 'T_OR', 'T_DR', 'T_Elo', 'O_Elo']].merge(games.reset_index(), left_on=['TID', 'OID', 'DayNum'], right_on=['WTeamID', 'LTeamID', 'DayNum'])
sdf2 = sdf[['DayNum', 'T_Poss', 'T_Ast', 'T_Score', 'T_OR', 'T_DR', 'T_Elo', 'O_Elo']].merge(games.reset_index(), left_on=['OID', 'TID', 'DayNum'], right_on=['WTeamID', 'LTeamID', 'DayNum'])
sdf = sdf1.rename(columns={'WTeamID': 'TID'}).set_index(['GameID', 'TID']).append( \
    sdf2.rename(columns={'LTeamID': 'TID'}).set_index(['GameID', 'TID'])).drop( \
    columns=['DayNum', 'WTeamID', 'LTeamID']).sort_index()
#%%
#av_df = av_df.loc[av_df['Mins'] > 18]
av_df['MinPerc'] = np.digitize(av_df['Mins'], 
                                np.concatenate(([0], np.percentile(av_df['Mins'], [25, 50, 75]),
                                                [av_df['Mins'].max() + 1])))
# av_df['Exp'] = av_df.index.get_level_values(0)
# av_df['Exp'] = av_df['Exp'].groupby(['PlayerID']).apply(lambda x: abs(x - x.min()))
ov_perc = (av_df - av_df.mean()) / av_df.std()
for idx, grp in av_df.groupby(['Season']):
    ov_perc.loc[grp.index] = (grp - grp.mean()) / grp.std()
ov_perc = ov_perc.drop(columns=['Mins', 'MinPerc'])

# sdf = fl.arrangeFrame(files, scaling=None, noinfluence=True)[0].loc(axis=0)[:, season, :, :]
# savdf = st.getSeasonalStats(sdf, strat='relelo')
#%%
    
print('Running scoring algorithms...')
ov_perc[['FoulPer18', 'TOPer18']] = -ov_perc[['FoulPer18', 'TOPer18']]
off_cons = np.array([1.2, 1.2, 0, 0, 1, 0, .7, 0, 0, .2, .2, 0, 0, 1, 0, 1, .5, 1, .8])
def_cons = np.array([0, 0, 1.5, 1.5, 0, 1.2, 0, .9, .6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
adj_mins = av_df['Mins']
adj_mins = adj_mins.loc[ov_perc.index]
aug_df = pd.DataFrame(index=av_df.index)

aug_df['OffScore'] = np.sum(ov_perc * off_cons, axis=1) / sum(off_cons)
aug_df['DefScore'] = np.sum(ov_perc * def_cons, axis=1) / sum(def_cons)
aug_df['2WayScore'] = aug_df['OffScore'] + aug_df['DefScore']
aug_df['HighScore'] = np.max(ov_perc, axis=1) - np.mean(ov_perc, axis=1)
aug_df['LowScore'] = np.min(ov_perc, axis=1) - np.mean(ov_perc, axis=1)
#av_df['Mins'] = adj_mins * 40

#CLUSTERING TO FIND PLAYER TYPES

for n in range(1, 5):
    tmp_df = av_df.loc[av_df['MinPerc'] == n].drop(columns=['Mins', 'MinPerc'])
    print('Applying clustering...')
    n_types = 5
    
    #Get number of clusters that best distinguishes between players
    cluster_algs = [
                        cl.KMeans(n_clusters=n_types)
                    ]
    
    kpca = KernelPCA(n_components=5)
    pca_df = pd.DataFrame(index=tmp_df.index, data=kpca.fit_transform(tmp_df))
    cat_perc = pca_df.copy()
    for idx, c in enumerate(cluster_algs):
        cat = c.fit_predict(pca_df)
    for n in range(len(list(set(cluster_algs[-1].labels_)))):
        cptmp = pca_df.loc[cat == n]
        cat_perc.loc[cat == n] = (cptmp - cptmp.mean()) / cptmp.std()
    cat_nme = 'C{}'.format(idx)
    aug_df.loc[tmp_df.index, cat_nme] = cat
    aug_df.loc[tmp_df.index, 'CatScore'] = np.sum(cat_perc, axis=1)
    aug_df.loc[tmp_df.index, 'CatScore'] = \
        (aug_df.loc[tmp_df.index, 'CatScore'] - \
          aug_df.loc[tmp_df.index, 'CatScore'].mean()) / \
         aug_df.loc[tmp_df.index, 'CatScore'].std()
av_df = av_df.join(aug_df)
for idx, grp in av_df.groupby(['Season']):
    for col in av_df.columns:
        if 'Score' in col:
            av_df.loc[grp.index, col] = \
                ((grp[col] - \
                  grp[col].mean()) / \
                 grp[col].std()) * 15 + 50


#%%

print('Getting team scoring data...')
ts_cols = [col for col in av_df.columns if 'Score' in col] + \
    [col + 'W' for col in av_df.columns if 'Score' in col]
ts_df = pd.DataFrame(index=av_df.groupby(['Season', 'TeamID']).mean().index,
                     columns=ts_cols).astype(np.float)
for idx, grp in av_df.groupby(['Season', 'TeamID']):
    for col in av_df.columns:
        if 'Score' in col:
            ts_df.loc[idx, [col, col + 'W']] = [grp[col].mean(), np.average(grp[col], weights=grp['Mins'])]
    
#%%

#Lineup experimentation
g_ev, line, sec_df, l_df = ev.getSingleGameLineups(2545, files, season)

lin1 = pd.DataFrame(columns=av_df.columns)
lin2 = pd.DataFrame(columns=av_df.columns)
for l in line[0]:
    try:
        lin1.loc[l] = av_df.loc(axis=0)[season, line[0][l]].sum()
    except:
        print('Lineup contains non-impact player.')
for col in lin1.columns:
    if 'Per18' not in col:
        lin1[col] = lin1[col] / 5
for l in line[1]:
    try:
        lin2.loc[l] = av_df.loc(axis=0)[season, line[1][l]].sum()
    except:
        print('Lineup contains non-impact player.')
for col in lin2.columns:
    if 'Per18' not in col:
        lin2[col] = lin2[col] / 5



#%%
plt.close('all')
#Figures
plt_cols = ['OffScore', 'DefScore',
            'PtsPer18', 'Dunk%', cat_nme]
    







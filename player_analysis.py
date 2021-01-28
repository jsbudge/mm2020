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
from scipy.stats import iqr, norm
from collections import Counter
from sportsreference.ncaab.roster import Player
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
        
season = 2016
# av_df = pd.DataFrame()
# m_df = pd.DataFrame()
# pdf = ev.getTeamRosters(files)
# for seas in [2015, 2016, 2017, 2018, 2019, 2020]:
#     sdf = fl.arrangeFrame(files, season, scaling=None, noinfluence=True)[0]
#     games = pd.read_csv('./data/{}/Games{}.csv'.format(seas, seas)).set_index(['GameID'])
#     roster_df = pd.read_csv('./data/{}/Rosters{}.csv'.format(seas, seas)).set_index(['GameID', 'PlayerID'])
#     sdf1 = sdf[['DayNum', 'T_Poss', 'T_Ast', 'T_Score', 'T_OR', 'T_DR', 'T_Elo', 'O_Elo']].merge(games.reset_index(), left_on=['TID', 'OID', 'DayNum'], right_on=['WTeamID', 'LTeamID', 'DayNum'])
#     sdf2 = sdf[['DayNum', 'T_Poss', 'T_Ast', 'T_Score', 'T_OR', 'T_DR', 'T_Elo', 'O_Elo']].merge(games.reset_index(), left_on=['OID', 'TID', 'DayNum'], right_on=['WTeamID', 'LTeamID', 'DayNum'])
#     sdf = sdf1.rename(columns={'WTeamID': 'TID'}).set_index(['GameID', 'TID']).append( \
#         sdf2.rename(columns={'LTeamID': 'TID'}).set_index(['GameID', 'TID'])).drop( \
#         columns=['DayNum', 'WTeamID', 'LTeamID']).sort_index()
                                                                                   
#     scale_df = ev.getRateStats(roster_df, sdf, pdf)
#     roster_df = roster_df.merge(scale_df, on=['GameID', 'PlayerID'])
#     #games, roster_df = ev.getRosters(files, seas)
#     games.to_csv('./data/{}/Games{}.csv'.format(seas, seas))
#     roster_df.to_csv('./data/{}/Rosters{}.csv'.format(seas, seas))
#     sum_df = roster_df.groupby(['PlayerID']).sum()
#     tmp_df = ev.getAdvStats(sum_df)
#     tmp_df['Season'] = seas
#     tmp_df['Mins'] = sum_df['Mins']
#     tmp_df = tmp_df.reset_index().set_index(['Season', 'PlayerID'])
#     av_df = av_df.append(tmp_df)
#     tmp_df = roster_df.groupby(['PlayerID']).mean()
#     tmp_df['Season'] = seas
#     tmp_df = tmp_df.reset_index().set_index(['Season', 'PlayerID'])
#     m_df = m_df.append(tmp_df)
# av_df = av_df.merge(pdf['TeamID'], on='PlayerID', right_index=True)
# av_df = av_df.rename(columns={'TeamID': 'TID'})
# av_df = av_df.reset_index().set_index(['Season', 'PlayerID', 'TID'])
# m_df = m_df.merge(pdf['TeamID'], on='PlayerID', right_index=True)
# m_df = m_df.rename(columns={'TeamID': 'TID'})
# m_df = m_df.reset_index().set_index(['Season', 'PlayerID', 'TID'])
# av_df.to_csv('./data/AVRosterData.csv')
# m_df.to_csv('./data/MeanRosterData.csv')
print('Loading player data...')
games = pd.read_csv('./data/{}/Games{}.csv'.format(season, season)).set_index(['GameID'])
roster_df = pd.read_csv('./data/{}/Rosters{}.csv'.format(season, season)).set_index(['GameID', 'PlayerID'])
sdf = fl.arrangeFrame(files, scaling=None, noinfluence=True)[0]
savdf = st.getSeasonalStats(sdf, strat='relelo')
tdf = fl.arrangeTourneyGames(files)[0]
adv_tdf = st.getTourneyStats(tdf, sdf, files) 
pdf = ev.getTeamRosters(files)
#%%

av_df = pd.read_csv('./data/InternetPlayerData.csv').set_index(['Season', 'PlayerID', 'TID']).sort_index()
av_df = av_df.loc[np.logical_not(av_df.index.duplicated())]
av_df = av_df.loc(axis=0)[:2019, :, :]
adv_df, phys_df, score_df, base_df = ev.splitStats(av_df, savdf)

ov_perc = adv_df.copy()
for idx, grp in ov_perc.groupby(['Season']):
    ov_perc.loc[grp.index] = (grp - grp.mean()) / grp.std()
ov_perc = ov_perc.sort_index()
#%%
    
print('Running scoring algorithms...')
n_clusters = 15
#ov_perc[['FoulPer18', 'TOPer18']] = -ov_perc[['FoulPer18', 'TOPer18']]
stat_cats = KernelPCA(n_components=n_clusters)
merge_df = ov_perc #pd.DataFrame(data=stat_cats.fit_transform(ov_perc), index=ov_perc.index)
#merges = [[col for n, col in enumerate(ov_perc.columns) if stat_cats.labels_[n] == q] for q in range(n_clusters)]
for idx, grp in merge_df.groupby(['Season']):
    merge_df.loc[grp.index] = (grp - grp.mean()) / grp.std()
m_cov = merge_df.merge(savdf[['T_OffRat', 'T_DefRat']], on=['Season', 
                                   'TID'], right_index=True).cov()
cov_cols = [col for col in m_cov.columns if 'T_' not in col]
off_cons = m_cov.loc['T_OffRat', cov_cols].values
def_cons = m_cov.loc['T_DefRat', cov_cols].values
adj_mins = phys_df['Mins']
adj_mins = adj_mins.loc[merge_df.index]
aug_df = pd.DataFrame(index=adv_df.index)

aug_df['OffScore'] = np.sum(merge_df * off_cons, axis=1) / sum(off_cons)
aug_df['DefScore'] = np.sum(merge_df * def_cons, axis=1) / sum(def_cons)
aug_df['2WayScore'] = 1 / abs(aug_df['OffScore'] - aug_df['DefScore'])**(1/2)
aug_df['StrengthScore'] = np.max(merge_df, axis=1) - np.mean(merge_df, axis=1)
aug_df['WeakScore'] = abs(np.min(merge_df, axis=1) - np.mean(merge_df, axis=1))
aug_df['BalanceScore'] = np.sqrt(aug_df['StrengthScore']**2 + aug_df['WeakScore']**2)
#adv_df['Mins'] = adj_mins * 40

#CLUSTERING TO FIND PLAYER TYPES
kpca = KernelPCA(n_components=3)

for n in range(1, 5):
    tmp_df = merge_df.loc[phys_df['MinPerc'] == n]
    pca_df = pd.DataFrame(index=tmp_df.index, data=kpca.fit_transform(tmp_df))
    cat_perc = pca_df.copy()
    print('Applying clustering...')
    n_types = 5
    clalg = cl.Birch(n_clusters=n_types, threshold=0.3755102040816327, branching_factor=94)
    cat = clalg.fit_predict(pca_df)
    shape_sz = pca_df.shape[0]
    big_cat = Counter(cat)
    #Get number of clusters that best distinguishes between players
    def min_func(x):
        clalg = cl.Birch(n_clusters=n_types, threshold=x[0], branching_factor=int(x[1]))
        cat = clalg.fit_predict(pca_df)
        big_cat = Counter(cat)
        return np.linalg.norm([big_cat[t] - shape_sz for t in big_cat])
    
    for n in range(len(list(set(clalg.labels_)))):
        cptmp = pca_df.loc[cat == n]
        cat_perc.loc[cat == n] = (cptmp - cptmp.mean()) / cptmp.std()
    aug_df.loc[tmp_df.index, 'Cat'] = cat
    aug_df.loc[tmp_df.index, 'CatScore'] = np.sum(cat_perc, axis=1)
    aug_df.loc[tmp_df.index, 'CatScore'] = \
        norm.cdf((aug_df.loc[tmp_df.index, 'CatScore'] - \
          aug_df.loc[tmp_df.index, 'CatScore'].mean()) / \
         aug_df.loc[tmp_df.index, 'CatScore'].std()) * 100
adv_df = adv_df.join(aug_df)
for idx, grp in adv_df.groupby(['Season']):
    for col in adv_df.columns:
        if 'Score' in col and 'Cat' not in col:
            adv_df.loc[grp.index, col] = \
                norm.cdf(((grp[col] - \
                  grp[col].mean()) / \
                 grp[col].std())) * 100


#%%

print('Getting team scoring data...')
ts_cols = [col for col in adv_df.columns if 'Score' in col] + \
    [col + 'W' for col in adv_df.columns if 'Score' in col]
ts_df = pd.DataFrame(index=adv_df.groupby(['Season', 'TID']).mean().index,
                     columns=ts_cols).astype(np.float)
for idx, grp in adv_df.groupby(['Season', 'TID']):
    for col in adv_df.columns:
        if 'Score' in col:
            ts_df.loc[idx, [col, col + 'W']] = [grp[col].mean(), np.average(grp[col], 
                                                                            weights=phys_df.loc[grp.index, 'MinsPerGame'].values)]
    
#%%

#Lineup experimentation
g_ev, line, sec_df, l_df = ev.getSingleGameLineups(2545, files, season)

lin1 = pd.DataFrame(columns=adv_df.columns)
lin2 = pd.DataFrame(columns=adv_df.columns)
for l in line[0]:
    try:
        lin1.loc[l] = adv_df.loc(axis=0)[season, line[0][l]].sum()
    except:
        print('Lineup contains non-impact player.')
for col in lin1.columns:
    if 'Per18' not in col:
        lin1[col] = lin1[col] / 5
for l in line[1]:
    try:
        lin2.loc[l] = adv_df.loc(axis=0)[season, line[1][l]].sum()
    except:
        print('Lineup contains non-impact player.')
for col in lin2.columns:
    if 'Per18' not in col:
        lin2[col] = lin2[col] / 5

#%%
plt_df = adv_df.loc[phys_df['MinPerc'] == 4, ['Cat', 'DWS', 'OWS',
                               'DefScore', 'OffScore', '2WayScore']]
plt_df = plt_df.loc(axis=0)[season, :, :]
plt_df = plt_df.loc[plt_df['Cat'] != -1]
pg = sns.PairGrid(plt_df, hue='Cat', palette=sns.husl_palette(len(list(set(plt_df['Cat'])))))
pg.map_lower(sns.scatterplot)
pg.map_diag(sns.kdeplot)

#%%
tourn_df = ts_df.merge(adv_tdf, on=['Season', 'TID'])[['OffScoreW', 'DefScoreW', 'T_RoundRank', 'T_FinalElo', 'T_Seed', 'CatScoreW']]
pg = sns.PairGrid(tourn_df, hue='T_RoundRank', palette=sns.husl_palette(len(list(set(tourn_df['T_RoundRank'])))))
pg.map_lower(sns.scatterplot, size=tourn_df['T_RoundRank'])
pg.map_diag(sns.kdeplot)

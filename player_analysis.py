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
#     #sdf = fl.arrangeFrame(files, season, scaling=None, noinfluence=True)[0]
#     games = pd.read_csv('./data/{}/Games{}.csv'.format(seas, seas)).set_index(['GameID'])
#     rosters = pd.read_csv('./data/{}/Rosters{}.csv'.format(seas, seas)).set_index(['GameID', 'PlayerID'])
#     #games, rosters = ev.getRosters(files, seas)
#     #games.to_csv('./data/{}/Games{}.csv'.format(seas, seas))
#     #rosters.to_csv('./data/{}/Rosters{}.csv'.format(seas, seas))
#     sum_df = rosters.groupby(['PlayerID']).sum()
#     tmp_df = ev.getAdvStats(sum_df)
#     tmp_df['Season'] = seas
#     tmp_df['Mins'] = sum_df['Mins']
#     tmp_df = tmp_df.reset_index().set_index(['Season', 'PlayerID'])
#     av_df = av_df.append(tmp_df)
#     tmp_df = rosters.groupby(['PlayerID']).mean()
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
#games, roster_df = ev.getRosters(files, season)
#%%
pdf = ev.getTeamRosters(files)
av_df = pd.read_csv('./data/AVRosterData.csv').set_index(['Season', 'PlayerID', 'TID'])
m_df = pd.read_csv('./data/MeanRosterData.csv').set_index(['Season', 'PlayerID', 'TID'])

sdf = fl.arrangeFrame(files, season, scaling=None, noinfluence=True)[0]
sdf1 = sdf[['DayNum', 'T_Poss', 'T_Ast', 'T_Score', 'T_OR', 'T_DR', 'T_Elo', 'O_Elo']].merge(games.reset_index(), left_on=['TID', 'OID', 'DayNum'], right_on=['WTeamID', 'LTeamID', 'DayNum'])
sdf2 = sdf[['DayNum', 'T_Poss', 'T_Ast', 'T_Score', 'T_OR', 'T_DR', 'T_Elo', 'O_Elo']].merge(games.reset_index(), left_on=['OID', 'TID', 'DayNum'], right_on=['WTeamID', 'LTeamID', 'DayNum'])
sdf = sdf1.rename(columns={'WTeamID': 'TID'}).set_index(['GameID', 'TID']).append( \
    sdf2.rename(columns={'LTeamID': 'TID'}).set_index(['GameID', 'TID'])).drop( \
    columns=['DayNum', 'WTeamID', 'LTeamID']).sort_index()
                                                                               
scale_df = ev.getRateStats(roster_df, sdf, pdf)
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
ov_perc = ov_perc.drop(columns=['Mins', 'MinPerc']).sort_index()

# sdf = fl.arrangeFrame(files, scaling=None, noinfluence=True)[0].loc(axis=0)[:, season, :, :]
# savdf = st.getSeasonalStats(sdf, strat='relelo')
#%%
    
print('Running scoring algorithms...')
ov_perc[['FoulPer18', 'TOPer18']] = -ov_perc[['FoulPer18', 'TOPer18']]
off_cons = np.array([1.1, 1.1, 0, 0, 1, 0, .7, 0, 0, .2, .2, 0, 0, 1, 0, 1, .5, 1, .8])
def_cons = np.array([0, 0, 1.4, 1.4, 0, 1.3, 0, 1, .3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
adj_mins = av_df['Mins']
adj_mins = adj_mins.loc[ov_perc.index]
aug_df = pd.DataFrame(index=av_df.index)

aug_df['OffScore'] = np.sum(ov_perc * off_cons, axis=1) / sum(off_cons)
aug_df['DefScore'] = np.sum(ov_perc * def_cons, axis=1) / sum(def_cons)
aug_df['2WayScore'] = 1 / abs(aug_df['OffScore'] - aug_df['DefScore'])**(1/40) + (aug_df['OffScore'] + aug_df['DefScore'])
aug_df['StrengthScore'] = np.max(ov_perc, axis=1) - np.mean(ov_perc, axis=1)
aug_df['WeakScore'] = abs(np.min(ov_perc, axis=1) - np.mean(ov_perc, axis=1))
aug_df['BalanceScore'] = np.sqrt(aug_df['StrengthScore']**2 + aug_df['WeakScore']**2)
#av_df['Mins'] = adj_mins * 40

#CLUSTERING TO FIND PLAYER TYPES
kpca = KernelPCA(n_components=3)

for n in range(1, 5):
    tmp_df = ov_perc.loc[av_df['MinPerc'] == n]
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
av_df = av_df.join(aug_df)
for idx, grp in av_df.groupby(['Season']):
    for col in av_df.columns:
        if 'Score' in col and 'Cat' not in col:
            av_df.loc[grp.index, col] = \
                norm.cdf(((grp[col] - \
                  grp[col].mean()) / \
                 grp[col].std())) * 100


#%%

print('Getting team scoring data...')
ts_cols = [col for col in av_df.columns if 'Score' in col] + \
    [col + 'W' for col in av_df.columns if 'Score' in col]
ts_df = pd.DataFrame(index=av_df.groupby(['Season', 'TID']).mean().index,
                     columns=ts_cols).astype(np.float)
for idx, grp in av_df.groupby(['Season', 'TID']):
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
av_check = ov_perc.merge(av_df[['MinPerc', 'Cat']], on=['Season', 'PlayerID', 'TID'])
av_cat = av_check.groupby(['MinPerc', 'Cat']).mean()

av_check = av_check.loc[av_check['MinPerc'] == 4].drop(columns=['MinPerc', 'Cat'])
av_cat = av_cat.loc(axis=0)[4, :]
ex_player = {}
for idx, row in av_cat.iterrows():
    dists = np.sqrt(np.sum((av_check - row)**2, axis=1))
    ex_player[idx[1]] = pdf.loc[dists.loc[dists == dists.min()].index.get_level_values(1).values[0],
                                'FullName']
    
prof = av_df.groupby(['MinPerc', 'Cat']).mean()
prof = prof.loc(axis=0)[4, :]
prof['ExPlayer'] = [ex_player[p] for p in ex_player if p in prof.index.get_level_values(1)]

#%%
plt_df = av_df.loc[av_df['MinPerc'] == 4, ['Cat', 'eFG%',
                               'DefScore', 'Econ']]
plt_df = plt_df.loc(axis=0)[season, :, :].merge(scale_df.groupby('PlayerID').mean(), on='PlayerID', right_index=True)
plt_df = plt_df.loc[plt_df['Cat'] != -1]
pg = sns.PairGrid(plt_df, hue='Cat', palette=sns.husl_palette(len(list(set(plt_df['Cat'])))))
pg.map_lower(sns.scatterplot)
pg.map_diag(sns.kdeplot)
#sns.pairplot(plt_df, hue='Cat', diag_kind=None, palette=sns.husl_palette(len(list(set(plt_df['Cat'])))))

av_seas = av_df.loc(axis=0)[season, :, :].merge(scale_df.groupby('PlayerID').mean(), on='PlayerID', right_index=True)
av_seas = av_seas.loc[av_seas['MinPerc'] == 4]


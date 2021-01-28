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
from sklearn.decomposition import TruncatedSVD, KernelPCA
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

n_clusters = 20
n_kpca_comps = 6
n_player_types = 5
minbins = np.array([0, 181, 456, 1160, 1649])
av_df = pd.read_csv('./data/InternetPlayerData.csv').set_index(['Season', 'PlayerID', 'TID']).sort_index()
av_df = av_df.loc[np.logical_not(av_df.index.duplicated())]
av_df = av_df.loc(axis=0)[:2019, :, :]
adv_df, phys_df, score_df, base_df = ev.splitStats(av_df, savdf, add_stats=['poss'],
                                                   minbins = minbins)

ov_perc = adv_df.copy()
for idx, grp in ov_perc.groupby(['Season']):
    ov_perc.loc[grp.index] = (grp - grp.mean()) / grp.std()
ov_perc = ov_perc.sort_index()
#%%
    
print('Running scoring algorithms...')

#ov_perc[['FoulPer18', 'TOPer18']] = -ov_perc[['FoulPer18', 'TOPer18']]
stat_cats = TruncatedSVD(n_components=n_clusters)
merge_df = pd.DataFrame(data=stat_cats.fit_transform(ov_perc.drop(columns=['SoS'])),
                        columns=['comp_{}'.format(n) for n in range(n_clusters)],
                        index=ov_perc.index)
for idx, grp in merge_df.groupby(['Season']):
    merge_df.loc[grp.index] = (grp - grp.mean()) / grp.std()
m_cov = merge_df.merge(savdf[['T_OffRat', 'T_DefRat']], on=['Season', 
                                    'TID'], right_index=True).cov()
cov_cols = [col for col in m_cov.columns if 'T_' not in col]
shift_cons = m_cov.loc['T_OffRat', cov_cols].values
off_cons = m_cov.loc['T_OffRat', cov_cols].values
def_cons = m_cov.loc['T_DefRat', cov_cols].values
shift_cons[abs(off_cons) - abs(def_cons) < 0] = 1
shift_cons[abs(def_cons) - abs(off_cons) < 0] = -1
off_cons[shift_cons == 1] = 0
def_cons[shift_cons == -1] = 0
aug_df = pd.DataFrame(index=adv_df.index)

aug_df['OffScore'] = np.sum(merge_df * off_cons, axis=1) / abs(sum(off_cons))
aug_df['DefScore'] = np.sum(merge_df * def_cons, axis=1) / sum(def_cons)
aug_df['OverallScore'] = (aug_df['OffScore'] + aug_df['DefScore'] + 3) * adv_df['SoS']
aug_df['BalanceScore'] = 1 / abs(aug_df['OffScore'] - aug_df['DefScore'])**(1/2)

#CLUSTERING TO FIND PLAYER TYPES
kpca = KernelPCA(n_components=n_kpca_comps)

for min_class in list(set(phys_df['MinPerc'].values)):
    tmp_df = merge_df.loc[phys_df['MinPerc'] == min_class]
    pca_df = pd.DataFrame(index=tmp_df.index, data=kpca.fit_transform(tmp_df))
    cat_perc = pca_df.copy()
    print('Applying clustering...')
    n_types = n_player_types
    clalg = cl.Birch(n_clusters=n_player_types, threshold=0.3755102040816327, branching_factor=94)
    cat = clalg.fit_predict(pca_df)
    shape_sz = pca_df.shape[0]
    big_cat = Counter(cat)
    #Get number of clusters that best distinguishes between players
    def min_func(x):
        clalg = cl.Birch(n_clusters=n_player_types, threshold=x[0], branching_factor=int(x[1]))
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
#Some example players from each segment of MinPerc

exdf = adv_df.merge(phys_df['MinPerc'], on=['Season', 'PlayerID', 'TID']).reset_index()
ex_players = pd.DataFrame()
best_players = pd.DataFrame()
for idx, grp in exdf.groupby(['MinPerc', 'Cat']):
    vals = grp.drop(columns = [col for col in grp.columns if 'Score' in col] + \
                    ['Season', 'PlayerID', 'TID', 'Cat', 'MinPerc'])
    vals = (vals - vals.mean()) / vals.std()
    dists = np.linalg.norm(vals, axis=1)
    player = grp.loc[dists == dists.min()]
    if player['PlayerID'].values[0] > 0:
        pid = pdf.loc[player['PlayerID'], 'FullName'].values[0]
        ex_players = ex_players.append(pd.DataFrame(list(player.values.flatten()) + \
                                                    [pid]).T)
    player = grp.loc[grp['CatScore'] == grp['CatScore'].max()]
    if player['PlayerID'].values[0] > 0:
        pid = pdf.loc[player['PlayerID'], 'FullName'].values[0]
        best_players = best_players.append(pd.DataFrame(list(player.values.flatten()) + \
                                                    [pid]).T)
ex_players.columns = list(exdf.columns) + ['PlayerName']
ex_players = ex_players.set_index(['MinPerc', 'Cat'])
best_players.columns = list(exdf.columns) + ['PlayerName']
best_players = best_players.set_index(['MinPerc', 'Cat'])
    
        
#%%

plt_df = adv_df.loc[phys_df['MinPerc'] == 3, ['Cat', 'DWS', 'OWS',
                               'DefScore', 'OffScore', 'BalanceScore']]
plt_df = plt_df.loc(axis=0)[season, :, :]
plt_df = plt_df.loc[plt_df['Cat'] != -1]
pg = sns.PairGrid(plt_df, hue='Cat', palette=sns.husl_palette(len(list(set(plt_df['Cat'])))))
pg.map_lower(sns.scatterplot)
pg.map_diag(sns.kdeplot)

#%%

tourn_df = ts_df.merge(adv_tdf, on=['Season', 'TID'])[['T_FinalRank', 'DefScoreW', 'OffScoreW', 'OverallScoreW', 'T_RoundRank', 'T_FinalElo', 'T_Seed', 'CatScoreW']]
pg = sns.PairGrid(tourn_df, hue='T_RoundRank', palette=sns.husl_palette(len(list(set(tourn_df['T_RoundRank'])))))
pg.map_lower(sns.scatterplot, size=tourn_df['T_RoundRank'])
pg.map_diag(sns.kdeplot)

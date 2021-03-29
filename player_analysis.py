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

def scaleDF(df, scale):
    return pd.DataFrame(index=df.index, columns=df.columns,
                        data=scale.fit_transform(df))
        
save_to_csv = False
print('Loading player data...')
sdf = st.arrangeFrame(scaling=None, noinfluence=True)[0]
savdf = st.getSeasonalStats(sdf, strat='relelo')
tdf = st.arrangeTourneyGames()[0]
adv_tdf = st.getTourneyStats(tdf, sdf)
pdf = ev.getTeamRosters()
scale = PowerTransformer()
#%%

n_kpca_comps = 10
n_player_types = 5
minbins = np.array([0, 5.55, 15.88, 30, 44]) #Chosen by empirical evidence - looking at a minutes distribution
av_df = pd.read_csv('./data/InternetPlayerData.csv').set_index(['Season', 'PlayerID', 'TID']).sort_index()
av_df = av_df.loc[np.logical_not(av_df.index.duplicated())]
adv_df, phys_df, score_df, base_df = ev.splitStats(av_df, savdf, add_stats=['poss'],
                                                   minbins = minbins)
posnum = list(set(phys_df['Pos'].values))

#%%
ov_perc = adv_df.sort_index()
ov_perc = scaleDF(ov_perc, scale)
    
ov_perc = ov_perc.drop(columns='SoS')
#%%
    
print('Running scoring algorithms...')

#Removing the benchwarmers as they make the stats unstable
merge_df = ov_perc.loc[phys_df['MinPerc'] > 1] 

#This normalizes the stats so they're all on the same scale
merge_df = scaleDF(merge_df, scale)
    
#An attempt to create an all-in-one offense and defense score
m_cov = merge_df.join(savdf[['T_OffRat', 'T_DefRat']], on=['Season', 
                                    'TID']).cov()
cov_cols = [col for col in m_cov.columns if 'T_' not in col]
shift_cons = m_cov.loc['T_OffRat', cov_cols].values
off_cons = m_cov.loc['T_OffRat', cov_cols].values
def_cons = m_cov.loc['T_DefRat', cov_cols].values
shift_cons[(abs(off_cons) - abs(def_cons)) < 0] = 1
shift_cons[(abs(def_cons) - abs(off_cons)) < 0] = -1
off_cons[shift_cons == 1] = 0
def_cons[shift_cons == -1] = 0
aug_df = pd.DataFrame(index=adv_df.index)

aug_df['OffScore'] = np.sum(merge_df * off_cons, axis=1) / abs(sum(off_cons))
aug_df['DefScore'] = np.sum(merge_df * def_cons, axis=1) / sum(def_cons)
aug_df['OverallScore'] = (aug_df['OffScore'] + aug_df['DefScore'] + 3) * adv_df['SoS']
aug_df['BalanceScore'] = 1 / abs(aug_df['OffScore'] - aug_df['DefScore'])**(1/2)

#CLUSTERING TO FIND PLAYER TYPES
kpca = TruncatedSVD(n_components=n_kpca_comps)

for min_class in posnum:
    tmp_df = merge_df.loc[phys_df['Pos'] == min_class]
    pca_df = pd.DataFrame(index=tmp_df.index, data=kpca.fit_transform(tmp_df))
    cat_perc = pca_df.copy()
    
    print('Applying clustering...')
    clalg = cl.Birch(n_clusters=n_player_types, threshold=0.44, branching_factor=45)
    cat = clalg.fit_predict(pca_df)
    aug_df.loc[tmp_df.index, 'Cat'] = cat
    
#Set the scores from 0-100ish because that makes it more intuitive
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
valid_adv_df = adv_df.dropna()
ts_cols = [col for col in adv_df.columns if 'Score' in col] + \
    [col + 'W' for col in adv_df.columns if 'Score' in col]
ts_df = pd.DataFrame(index=adv_df.groupby(['Season', 'TID']).mean().index,
                     columns=ts_cols).astype(np.float)
for idx, grp in valid_adv_df.groupby(['Season', 'TID']):
    posgrp = phys_df.loc[grp.index, ['Pos', 'MinsPerGame', 'Height', 'Weight']]
    for col in valid_adv_df.columns:
        if 'Score' in col:
            ts_df.loc[idx, [col, col + 'W']] = [grp[col].mean(), 
                                                np.average(grp[col], 
                                                           weights=posgrp['MinsPerGame'].values)]
            ts_df.loc[idx, ['Min' + col, 'Max' + col]] = [grp[col].min(), 
                                                          grp[col].max()]
    ts_df.loc[idx, 'Height'] = np.average(posgrp['Height'].values, 
                                          weights=posgrp['MinsPerGame'].values)
    ts_df.loc[idx, 'Weight'] = np.average(posgrp['Weight'].values, 
                                          weights=posgrp['MinsPerGame'].values)
    bmi = phys_df.loc[grp.index, 'Weight'].values * .454 / (posgrp['Height'].values * .0254)**2
    ts_df.loc[idx, 'BMI'] = np.average(bmi, 
                                       weights=posgrp['MinsPerGame'].values)
    
    for pos in posnum:
        if np.any(posgrp['Pos'] == pos):
            ts_df.loc[idx, 'Pos{}Q'.format(int(pos))] = \
                np.average(grp.loc[posgrp['Pos'] == pos, 'OverallScore'].values, 
                           weights=posgrp.loc[posgrp['Pos'] == pos, 'MinsPerGame'].values)
        else:
            ts_df.loc[idx, 'Pos{}Q'.format(int(pos))] = 0
if save_to_csv:
    ts_df.to_csv('./data/PlayerAnalysisData.csv')
    
#%%
for season in np.arange(2012, 2022):
    spdf = valid_adv_df.loc(axis=0)[season, :, :]
    spdf = spdf.loc(axis=0)[:, spdf.index.get_level_values(1) > 0, :]
    bop = spdf.loc[spdf['OffScore'] == spdf['OffScore'].max()].index.get_level_values(1).values[0]
    bdp = spdf.loc[spdf['DefScore'] == spdf['DefScore'].max()].index.get_level_values(1).values[0]
    bbp = spdf.loc[spdf['BalanceScore'] == spdf['BalanceScore'].max()].index.get_level_values(1).values[0]
    bovp = spdf.loc[spdf['OverallScore'] == spdf['OverallScore'].max()].index.get_level_values(1).values[0]
    print('Best players for {}:\n'.format(season))
    print('Offensive:\t' + pdf.loc[bop, 'FullName'])
    print('Defensive:\t' + pdf.loc[bdp, 'FullName'])
    print('Balance:\t' + pdf.loc[bbp, 'FullName'])
    print('Overall:\t' + pdf.loc[bovp, 'FullName'])
    print('')
        
#%%

# plt_df = adv_df[['Cat', 'OverallScore', 'DWS', 'OWS', 'WEcon']]
# for pos in [1, 3, 5]:
#     plt_df = adv_df[['Cat', 'OverallScore', 'DWS', 'OWS', 'WEcon']].loc[phys_df['Pos'] == pos].dropna()
#     plt_df = plt_df.loc[plt_df['Cat'] != -1]
#     pg = sns.PairGrid(plt_df, hue='Cat', palette=sns.husl_palette(plt_df['Cat'].max()+1))
#     pg.map_lower(sns.scatterplot)
#     pg.map_diag(sns.kdeplot)
#     plt.title('Pos {}'.format(pos))

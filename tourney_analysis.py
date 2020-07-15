#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:05:31 2020

@author: jeff

Tourney Analysis script.

Looking at some of the teams involved in the tournament,
how they compare to the average team.

I wanted a more permanent scripting system for the different mathematical
forays that I'm running.

This adds together the mean data from the season for participant teams
and does some stuff to try and find a good overall team predictor function.
"""

import numpy as np
import pylab as plab
import pandas as pd
import statslib as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import HuberRegressor, Lars, ElasticNet, Lasso, SGDRegressor, TheilSenRegressor, \
    ARDRegression, LassoLars
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR
from sklearn.neural_network import MLPClassifier
from plotlib import PlotGenerator, showStat
from scipy.optimize import minimize

#Gather all of our files
split_yr = 2013
files = st.getFiles()
#ts = st.getGames(files['MRegularSeasonDetailedResults']).drop(columns=['NumOT'])
st.calcElo(st.getGames(files['MRegularSeasonDetailedResults'], split=True)[0])
# ts = st.addRanks(ts)
# ts = st.addElos(ts)
# ts = st.joinFrame(ts, st.getStats(ts))
# ts = st.joinFrame(ts, st.getInfluenceStats(ts))
# tt = st.getGames(files['MNCAATourneyDetailedResults'], split=True)
# ttu = st.getGames(files['MNCAATourneyDetailedResults']).sort_values('GameID')
# ttstats = st.getTourneyStats(ttu, files)
# for idx, row in ttstats.iterrows():
#     ttstats.loc[idx, 'T_FinalElo'] = ts.loc[np.logical_and(ts['Season'] == idx[0],
#                                                            ts['TID'] == idx[1]),
#                                             'T_Elo'].values[-1]
#     ttstats.loc[idx, 'T_FinalRank'] = ts.loc[np.logical_and(ts['Season'] == idx[0],
#                                                            ts['TID'] == idx[1]),
#                                             'T_Rank'].values[-1]
#     tt[0].loc[np.logical_and(tt[0]['Season'] == idx[0],
#                              tt[0]['TID'] == idx[1]), 'T_FinalElo'] = ts.loc[np.logical_and(ts['Season'] == idx[0],
#                                                                                          ts['TID'] == idx[1]),
#                                                                                          'T_Elo'].values[-1]
#     tt[0].loc[np.logical_and(tt[0]['Season'] == idx[0],
#                              tt[0]['OID'] == idx[1]), 'O_FinalElo'] = ts.loc[np.logical_and(ts['Season'] == idx[0],
#                                                                                          ts['TID'] == idx[1]),
#                                                                                          'T_Elo'].values[-1]
#     tt[0].loc[np.logical_and(tt[0]['Season'] == idx[0],
#                              tt[0]['TID'] == idx[1]), 'T_FinalRank'] = ts.loc[np.logical_and(ts['Season'] == idx[0],
#                                                                                          ts['TID'] == idx[1]),
#                                                                                          'T_Rank'].values[-1]
#     tt[0].loc[np.logical_and(tt[0]['Season'] == idx[0],
#                              tt[0]['OID'] == idx[1]), 'O_FinalRank'] = ts.loc[np.logical_and(ts['Season'] == idx[0],
#                                                                                          ts['TID'] == idx[1]),
#                                                                                          'T_Rank'].values[-1]

# sts = st.getSeasonalStats(ts, strat='rank')
# sts = sts.merge(ttstats[['T_FinalRank', 'T_FinalElo', 'T_Seed']], left_index=True, right_index=True)
# sts = st.normalizeToSeason(sts)

# #Build the feature vector
# ttw = tt[0][['GameID', 'Season', 'TID']].merge(sts,
#                                                on=['Season', 'TID']).set_index(['GameID'])
# ttl = tt[1][['GameID', 'Season', 'TID']].merge(sts,
#                                                on=['Season', 'TID']).set_index(['GameID'])
# features = (ttw - ttl).drop(columns=['Season', 'TID'])
# features = features.append(-features, ignore_index=True)
# y = np.concatenate((np.ones((tt[0].shape[0],)),
#                     np.zeros((tt[1].shape[0],))))

# #Feature selection, to remove redundancies, etc.
# feat_select = SelectKBest(score_func=mutual_info_classif, k=50).fit(features, y)
# nndf = pd.DataFrame(feat_select.transform(features), columns=features.columns[feat_select.get_support()])

# #Load in all the sklearn stuff
# #kpca = KernelPCA(n_components=20)
# #clusterdf = pd.DataFrame(data=kpca.fit_transform(features))
# cv = StratifiedKFold(n_splits=5, shuffle=True)
# nn = MLPClassifier(hidden_layer_sizes=(500, 500, 500, 500, 250, 100,))
# for n, (train, test) in enumerate(cv.split(nndf, y)):
#     nn.fit(nndf.iloc[train], y[train])
#     print('Fold {}: {:.2f}'.format(n, nn.score(nndf.iloc[test], y[test])))
    
# print('Comparisons to metrics:')
# print('Elo: {:.2f}%'.format(sum(tt[0]['T_FinalElo'] > tt[0]['O_FinalElo']) / tt[0].shape[0] * 100))
# print('Rank: {:.2f}%'.format(sum(tt[0]['T_FinalRank'] < tt[0]['O_FinalRank']) / tt[0].shape[0] * 100))

runner = st.getGames(files['MRegularSeasonDetailedResults'], split=True)[0]
def minElos(x):
    wdf = st.calcElo(runner, x[0], x[1])
    wdf = wdf.loc[wdf.duplicated('GameID')]
    res = sum(wdf['T_Elo'] < wdf['O_Elo']) / wdf.shape[0]
    print(res)
    return 1 - res
test = minimize(minElos, [20, 2])

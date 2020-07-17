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
from sklearn.decomposition import KernelPCA, FastICA
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif, f_classif, \
    SelectPercentile, VarianceThreshold
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import Isomap
from sklearn.metrics import log_loss, make_scorer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from plotlib import PlotGenerator, showStat
from scipy.optimize import minimize
from tourney import Bracket

#Gather all of our files
split_yr = 2018
files = st.getFiles()
tnames = st.loadTeamNames(files)
ts = st.getGames(files['MRegularSeasonDetailedResults']).drop(columns=['NumOT'])
ts = st.addRanks(ts)
ts = st.addElos(ts)
ts = st.joinFrame(ts, st.getStats(ts))
ts = st.joinFrame(ts, st.getInfluenceStats(ts, recalc=True))
tt = st.getGames(files['MNCAATourneyDetailedResults'], split=True)
ttu = st.getGames(files['MNCAATourneyDetailedResults'])
ttstats = st.getTourneyStats(ttu, ts, files)

sts = st.getSeasonalStats(ts, strat='rank', recalc=True)
sts = sts.merge(ttstats[['T_FinalRank', 'T_FinalElo', 'T_Seed']], left_index=True, right_index=True)
sts = st.normalizeToSeason(sts, scaler=PowerTransformer())

#Build the feature vector
ttw = tt[0][['GameID', 'Season', 'TID']].merge(sts,
                                                on=['Season', 'TID']).set_index(['GameID'])
ttl = tt[1][['GameID', 'Season', 'TID']].merge(sts,
                                                on=['Season', 'TID']).set_index(['GameID'])
features = (ttw - ttl).drop(columns=['Season', 'TID'])
features = features.append(-features, ignore_index=True)
winner = np.concatenate((np.ones((tt[0].shape[0],)),
                    np.zeros((tt[1].shape[0],))))

#%%
#Feature selection, to remove redundancies, etc.
minfo_select = SelectPercentile(score_func=mutual_info_classif, percentile=25)
fclass_select = SelectPercentile(score_func=f_classif, percentile=35)
kpca_lin = KernelPCA(n_components=15, kernel='linear')
kpca_rbf = KernelPCA(n_components=15, kernel='rbf')
transform = FeatureUnion([('kpca_lin', kpca_lin),
                                              ('kpca_rbf', kpca_rbf),
                             ('scores', Pipeline([('minfo', minfo_select),
                             ('fclass', fclass_select)]))])

print('Fitting features...')
nfeats = pd.DataFrame(transform.fit_transform(features, winner))
#nfeats['GameID'] = ttu['GameID']

#Load in all the sklearn stuff
#clusterdf = pd.DataFrame(data=kpca.fit_transform(features))
cv = StratifiedKFold(n_splits=3, shuffle=True)
#nn = MLPClassifier()
rforest = RandomForestClassifier()

print('Running RFE...')
#rfe = RFECV(estimator=rforest, cv=cv).fit(nfeats, winner)

Xt = nfeats.loc[ttu['Season'] <= split_yr]; yt = winner[ttu['Season'] <= split_yr]
Xs = nfeats.loc[ttu['Season'] > split_yr]; ys = winner[ttu['Season'] > split_yr]
ys_ll = OneHotEncoder(sparse=False).fit_transform(ys.reshape(-1, 1))

#rf_pipe = Pipeline([('s1', transform), ('s2', rforest)])

#%%
print('Running grid search...')
rf_final = GridSearchCV(rforest,
                        param_grid={'n_estimators': [100, 300, 450, 500, 550, 700, 1000, 1500],
                                    'criterion': ['gini', 'entropy']},
                        cv=cv)
rf_final.fit(Xt, yt)

print('Log Losses')
#print('NN: {:.2f}'.format(log_loss(ys_ll, nn_final.predict_proba(Xs))))
print('RF: {:.2f}'.format(log_loss(ys_ll, rf_final.predict_proba(Xs))))

view = ttu.loc[ttu['Season'] > split_yr, ['GameID', 'Season', 'TID', 'OID']]
view['T%'] = (np.round(rf_final.predict_proba(Xs)[:, 1], 2) * 100).astype(np.int)
view['O%'] = (np.round(rf_final.predict_proba(Xs)[:, 0], 2) * 100).astype(np.int)
v2 = ((view.loc[view.duplicated(['GameID'], keep='first')] + view.loc[view.duplicated(['GameID'], keep='first')]) / 2).astype(np.int)
v2['TName'] = [tnames[tid] for tid in v2['TID']]
v2['OName'] = [tnames[tid] for tid in v2['OID']]

bracket = Bracket(2019, files)
testdf = sts.loc(axis=0)[2019, :]
testdf = testdf.reset_index().set_index('TID').drop(columns=['Season'])
bracket.run(testdf, rf_final, transformer=transform)
print('Score: {}'.format(bracket.score))
bracket.printTree('./test.txt')

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
from tourney import Bracket, FeatureCreator

#Gather all of our files
# split_yr = 2018
files = st.getFiles()
tt = st.getGames(files['MNCAATourneyDetailedResults'], split=True)
ttu = st.getGames(files['MNCAATourneyDetailedResults'])

print('Loading features...')
fc = FeatureCreator(files, scaling=RobustScaler())
Xt, yt = fc.splitGames(ttu)

#%%
#Feature selection, to remove redundancies, etc.
minfo_select = SelectPercentile(score_func=mutual_info_classif, percentile=5)
fclass_select = SelectPercentile(score_func=f_classif, percentile=25)
kpca_lin = KernelPCA(n_components=20, kernel='linear')
kpca_rbf = KernelPCA(n_components=20, kernel='rbf')
transform = FeatureUnion([('kpca_lin', kpca_lin),
                          ('kpca_rbf', kpca_rbf),
                          ('minfo', minfo_select)])

print('Fitting features...')
nfeats = pd.DataFrame(transform.fit_transform(features, winner))
fc.setTransform(transform)
cv = StratifiedKFold(n_splits=3, shuffle=True)
#nn = MLPClassifier()
rforest = RandomForestClassifier()

print('Running RFE...')
#rfe = RFECV(estimator=rforest, cv=cv).fit(nfeats, winner)

Xt = nfeats; yt = winner
Xs = nfeats; ys = winner
ys_ll = OneHotEncoder(sparse=False).fit_transform(ys.reshape(-1, 1))

#rf_pipe = Pipeline([('s1', transform), ('s2', rforest)])

#%%
print('Running grid search...')
rf_final = GridSearchCV(rforest,
                        param_grid={'n_estimators': [200, 225, 250],
                                    'criterion': ['gini', 'entropy']},
                        cv=cv)
rf_final.fit(Xt, yt)

print('Log Losses')
#print('NN: {:.2f}'.format(log_loss(ys_ll, nn_final.predict_proba(Xs))))
print('RF: {:.2f}'.format(log_loss(ys_ll, rf_final.predict_proba(Xs))))

bracket = Bracket(2019, files)
bracket.run(rf_final, fc)
print('Score: {}'.format(bracket.flat_score))
bracket.printTree('./test.txt')

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
from sklearn.decomposition import KernelPCA, FastICA, LatentDirichletAllocation, DictionaryLearning, \
    TruncatedSVD
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif, f_classif, \
    SelectPercentile, VarianceThreshold
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.manifold import Isomap
from sklearn.covariance import EmpiricalCovariance, OAS
from sklearn.metrics import log_loss, make_scorer
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, \
    PolynomialFeatures
from plotlib import PlotGenerator, showStat, showCorrs
from scipy.optimize import minimize
from tourney import Bracket, FeatureCreator

#Gather all of our files
# split_yr = 2018
files = st.getFiles()
cv = StratifiedKFold(n_splits=3, shuffle=True)
ttu = st.getGames(files['MNCAATourneyDetailedResults'])

print('Loading features...')
fc = FeatureCreator(files, scaling=PowerTransformer(), strat='rank')
sts = fc.avdf.copy()
Xtrans, ytrans = fc.loadGames(ttu)

#%%

print('Fitting features...')


f_transform = FeatureUnion([('kpca', KernelPCA(n_components=30, kernel='cosine')),
                            ('linkpca', KernelPCA(n_components=20)),
                            ('select', Pipeline([('minfo', SelectPercentile(score_func=mutual_info_classif, percentile=50)),
                                                 ('fclass', SelectPercentile(score_func=f_classif, percentile=25))]))])
g_transform = Pipeline([('svd', TruncatedSVD(n_components=20)),
                        ('scale', RobustScaler())])

fc.loadAndTransformGames(ttu, f_transform, g_transform, fc.getGameRank().values)
#rf_pipe = Pipeline([('s1', transform), ('s2', rforest)])

#%%
print('Running grid search...')
for season in range(2015, 2020):
    Xt, yt, Xs, ys = fc.splitGames(split=season)
    ys_ll = OneHotEncoder(sparse=False).fit_transform(ys.reshape(-1, 1))
    classifier = GridSearchCV(RandomForestClassifier(), 
                              param_grid={'n_estimators':[500],
                                          'criterion':['gini']}, 
                              cv=cv)
    classifier.fit(Xt, yt)
    bracket = Bracket(season, files)
    bracket.run(classifier, fc)
    print('{}: {:.2f}, {:.2f}, {}'.format(season, bracket.loss, bracket.accuracy, bracket.espn_score))

bracket.printTree('./test.txt')
showCorrs(sts, fc.avdf)

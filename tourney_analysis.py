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
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR
from plotlib import PlotGenerator, showStat

#Gather all of our files
split_yr = 2013
files = st.getFiles()
ts = st.getGames(files['MRegularSeasonDetailedResults']).drop(columns=['NumOT'])
tt = st.getGames(files['MNCAATourneyDetailedResults'], split=True)
ts = st.addRanks(ts)
ts = st.addElos(ts)
ts = st.joinFrame(ts, st.getStats(ts))
ts = st.joinFrame(ts, st.getInfluenceStats(ts))
sts = st.normalizeToSeason(st.getSeasonalStats(ts, strat='elo'))

#Build the feature vector
ttw = tt[0][['GameID', 'Season', 'TID']].merge(st.getTeamStats(sts, av=True), on=['Season', 'TID']).set_index(['GameID'])
ttl = tt[1][['GameID', 'Season', 'TID']].merge(st.getTeamStats(sts, av=True), on=['Season', 'TID']).set_index(['GameID'])
features = (ttw - ttl).drop(columns=['Season', 'TID'])
features = features.append(-features, ignore_index=True)
y = np.concatenate((np.ones((tt[0].shape[0],)),
                    np.zeros((tt[1].shape[0],))))

#Load in all the sklearn stuff
kpca = KernelPCA(n_components=20)
clusterdf = pd.DataFrame(data=kpca.fit_transform(features))
cv = StratifiedKFold(n_splits=5, shuffle=True)
svm_est = SVR(kernel='linear')
feat_cull = RFECV(svm_est, cv=cv)
feat_cull.fit(clusterdf, y)


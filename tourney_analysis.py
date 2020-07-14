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
from plotlib import PlotGenerator, showStat

split_yr = 2013
files = st.getFiles()
ts = st.getGames(files['MRegularSeasonDetailedResults']).drop(columns=['NumOT'])
tt = st.getGames(files['MNCAATourneyDetailedResults']).drop(columns=['NumOT'])
ts = st.addRanks(ts)
ts = st.addElos(ts)
ts = st.joinFrame(ts, st.getStats(ts))
tte = pd.DataFrame(index=tt.groupby(['Season', 'TID']).mean().index)
sts = st.getSeasonalStats(ts, strat='mean')

ttsts = tte.merge(sts, left_index=True, right_index=True)
tspecstats = st.getTourneyStats(tt, ttsts)

ttnorm = st.normalizeToSeason(ttsts)
tsnorm = st.normalizeToSeason(sts)

kpca = KernelPCA(n_components=15, kernel='linear')
clusterdf = pd.DataFrame(index=ttnorm.index, data=kpca.fit_transform(ttnorm)) #ttnorm.copy()
plt.close('all')
nms = ['Huber', 'ThielSen', 'SGD', 'ARD']
f = plt.figure()
gd = gridspec.GridSpec(len(nms), 2, figure=f)
for n, lm in enumerate([HuberRegressor(max_iter=1000), TheilSenRegressor(), SGDRegressor(),
                        ARDRegression()]):
    lm.fit(clusterdf.loc(axis=0)[ttnorm.index.get_level_values(0) < split_yr, :],
              tspecstats['GameRank'][tspecstats.index.get_level_values(0) < split_yr])
    tspecstats[nms[n] + 'Rank'] = 0
    tspecstats[nms[n] + 'Rank'] = lm.predict(clusterdf)
    
    f.add_subplot(gd[n, 0])
    plt.title(nms[n])
    plt.scatter(tspecstats.loc[ttsts.index.get_level_values(0) >= split_yr, 'GameRank'],
                tspecstats.loc[ttsts.index.get_level_values(0) >= split_yr, nms[n] + 'Rank'])
    f.add_subplot(gd[n, 1])
    plt.title('Dist')
    for r in range(7):
        sns.distplot(tspecstats.loc[ttsts.index.get_level_values(0) >= split_yr].loc[tspecstats['GameRank'] == r, nms[n] + 'Rank'])
        
tspecstats['MeanRank'] = tspecstats[[n + 'Rank' for n in nms]].mean(axis=1).values
tspecstats['SumRank'] = tspecstats[[n + 'Rank' for n in nms]].sum(axis=1).values
tspecstats['MaxRank'] = tspecstats[[n + 'Rank' for n in nms]].max(axis=1).values
tspecstats['MinRank'] = tspecstats[[n + 'Rank' for n in nms]].min(axis=1).values
tspecstats['MedianRank'] = tspecstats[[n + 'Rank' for n in nms]].median(axis=1).values

f2 = plt.figure()
gd2 = gridspec.GridSpec(5, 2, figure=f2)
for n, lm in enumerate(['Mean', 'Sum', 'Max', 'Min', 'Median']):
    f2.add_subplot(gd2[n, 0])
    plt.title(lm)
    plt.scatter(tspecstats.loc[ttsts.index.get_level_values(0) >= split_yr, 'GameRank'],
                tspecstats.loc[ttsts.index.get_level_values(0) >= split_yr, lm + 'Rank'])
    f2.add_subplot(gd2[n, 1])
    plt.title('Dist')
    for r in range(7):
        sns.distplot(tspecstats.loc[ttsts.index.get_level_values(0) >= split_yr].loc[tspecstats['GameRank'] == r, lm + 'Rank'])

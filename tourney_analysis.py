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
from sklearn.linear_model import HuberRegressor, Lars, ElasticNet, Lasso, SGDRegressor, TheilSenRegressor, \
    ARDRegression, LassoLars
from plotlib import PlotGenerator, showStat

split_yr = 2013
files = st.getFiles()
ts = st.getGames(files['MRegularSeasonDetailedResults']).drop(columns=['NumOT'])
tt = st.getGames(files['MNCAATourneyDetailedResults']).drop(columns=['NumOT'])
ts = st.addRanks(ts)
ts = st.addElos(ts)
ts = st.addStats(ts)
ts2019 = ts.loc[ts['Season'] == 2019]
tt2019 = tt.loc[tt['Season'] == 2019]
tte = pd.DataFrame(index=tt.groupby(['Season', 'T_TeamID']).mean().index)
sts = st.getSeasonalStats(ts, strat='hmean')

ttsts = tte.merge(sts, left_index=True, right_index=True)
tspecstats = st.getTourneyStats(tt, ttsts)

ttnorm = st.normalizeToSeason(ttsts)
tsnorm = st.normalizeToSeason(sts)
plt.close('all')
nms = ['Huber', 'ThielSen', 'SGD', 'ARD']
for n, lm in enumerate([HuberRegressor(max_iter=1000), TheilSenRegressor(), SGDRegressor(),
                        ARDRegression()]):
    lm.fit(ttsts.loc(axis=0)[ttnorm.index.get_level_values(0) < split_yr, :],
              tspecstats['GameRank'][tspecstats.index.get_level_values(0) < split_yr])
    tspecstats[nms[n] + 'Rank'] = 0
    tspecstats[nms[n] + 'Rank'] = lm.predict(ttnorm)
    
    plt.figure(nms[n])
    plt.scatter(tspecstats.loc[ttsts.index.get_level_values(0) >= split_yr, 'GameRank'],
                tspecstats.loc[ttsts.index.get_level_values(0) >= split_yr, nms[n] + 'Rank'])
    plt.figure(nms[n] + '_dist')
    for r in range(7):
        sns.distplot(tspecstats.loc[ttsts.index.get_level_values(0) >= split_yr].loc[tspecstats['GameRank'] == r, nms[n] + 'Rank'])

final_reg = SGDRegressor(loss='huber')
final_reg.fit(tspecstats.loc[ttsts.index.get_level_values(0) < split_yr, 
                             [n + 'Rank' for n in nms]],
              tspecstats['GameRank'][tspecstats.index.get_level_values(0) < split_yr])
tspecstats['CompRank'] = 0
tspecstats.loc[ttsts.index.get_level_values(0) >= split_yr, 
               'CompRank'] = final_reg.predict(tspecstats.loc[ttsts.index.get_level_values(0) >= split_yr, 
                             [n + 'Rank' for n in nms]])
plt.figure('Composite')
plt.scatter(tspecstats.loc[ttsts.index.get_level_values(0) >= split_yr, 'GameRank'],
            tspecstats.loc[ttsts.index.get_level_values(0) >= split_yr, 'CompRank'])
plt.figure('Composite_dist')
for r in range(7):
    sns.distplot(tspecstats.loc[ttsts.index.get_level_values(0) >= split_yr].loc[tspecstats['GameRank'] == r, 'CompRank'])

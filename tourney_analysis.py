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
ts = st.joinFrame(ts, st.getInfluenceStats(ts))
sts = st.getSeasonalStats(ts, strat='hmean')
tte = pd.DataFrame(index=tt.groupby(['Season', 'TID']).mean().index)
ttsts = tte.merge(sts, left_index=True, right_index=True)
tspecstats = st.getTourneyStats(tt, ttsts, files)

ttnorm = st.normalizeToSeason(ttsts)
tsnorm = st.normalizeToSeason(sts)

ttcorr = ttnorm.merge(tspecstats, left_index=True, right_index=True).corr()

plt.figure()
plt.plot(ttcorr['AdjGameRank'])

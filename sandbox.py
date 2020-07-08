#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:31:29 2020

@author: jeff

Play script

Stuff here gets added to statslib later.
"""

import numpy as np
import pandas as pd
import statslib as st
import seaborn as sns
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import matplotlib.pyplot as plt

'''
STUFF TO THINK ABOUT

* Exponent on the Pythagorean Wins formula. Try with different ones, see which fits
the college basketball model the best

* Outlier potential. Some form of Mahalanobis using the covariance matrix
of stats across the whole season. Might be good at predicting well-rounded teams.

* Compute Elo ratings for teams, setting initial using the past season tournament
results with a regression to the mean

* Find some good defensive stats - relative to the rest of people, anyway.

'''


files = st.getFiles()
ts = st.getGames(files['MRegularSeasonDetailedResults'])
tt = st.getGames(files['MNCAATourneyDetailedResults'])
ts = st.addRanks(ts)
ts = st.addStats(ts)
tt = st.addStats(tt)
ts = st.addElos(ts)
#ts = st.normalizeToSeason(ts)
#tt = st.normalizeToSeason(tt)
ts2019 = ts.loc[ts['Season'] == 2019]
tt2019 = tt.loc[tt['Season'] == 2019]

for tmid in [1140, 1181, 1101, 1438]:
    plt.plot(ts2019.loc[ts2019['T_TeamID'] == tmid].corr()['O_Rank'])
# rmns = st.getSeasonalStats(ts2019, strat='rank')
# emns = st.getSeasonalStats(ts2019, strat='elo')
# hmns = st.getSeasonalStats(ts2019, strat='hmean')
# amns = st.getSeasonalStats(ts2019, strat='mest')
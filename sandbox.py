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

* Find some good defensive stats - relative to the rest of people, anyway.

* Everything to do with Events. Haven't looked at them much.


'''


files = st.getFiles()
ts = st.getGames(files['MRegularSeasonDetailedResults'])
ts2019 = ts.loc[ts['Season'] >= 2018]
sweights = st.getSystemWeights(ts2019, files)
ts2019 = st.getRanks(ts2019, files)
wdf_diffs = st.getDiffs(ts2019)
score = sum(np.logical_or(np.logical_and(wdf_diffs['Rank_diff'] < 0, 
                         wdf_diffs['Score_diff'] > 0),
                        np.logical_and(wdf_diffs['Rank_diff'] > 0, 
                         wdf_diffs['Score_diff'] < 0))) / wdf_diffs.shape[0]
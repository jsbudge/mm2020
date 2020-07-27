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
from plotlib import PlotGenerator, showStat
from sklearn.decomposition import KernelPCA
from sklearn.cluster import AffinityPropagation, DBSCAN
from tourney import Bracket, FeatureFrame, GameFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, SelectPercentile

'''
STUFF TO THINK ABOUT

* Exponent on the Pythagorean Wins formula. Try with different ones, see which fits
the college basketball model the best

* Outlier potential. Some form of Mahalanobis using the covariance matrix
of stats across the whole season. Might be good at predicting well-rounded teams.

* Find some good defensive stats - relative to the rest of people, anyway.

* Everything to do with Events. Haven't looked at them much.

* See if there's a way to quantify teams based on statistical profile -
for example, offensive vs. defensive or fast vs. slow. 

* Analyze shifts between tournament and regular season. Is there a way to predict
those?
'''

'''
This is the procedure to load all the rank, elo, weights, etc.
CSV files
'''
files = st.getFiles()
feats = FeatureFrame(files, scaling=StandardScaler())
#feats.add_t(('as', StandardScaler()))
print('Transforming features...')
feats.execute()
#other = feats.get(tid=1141)
print('Running GameFrame...')
test = GameFrame(files, feats)
test.add_t(('rs', RobustScaler()))
test.add_c(('rf', RandomForestClassifier()))
print('Executing...')
test.execute(*test.splitGames(2019))
#Xt, yt, Xs, ys = test.splitGames(2019)

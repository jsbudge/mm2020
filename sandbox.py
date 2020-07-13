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
ts = st.getGames(files['MRegularSeasonDetailedResults'])
ts_split = st.getGames(files['MRegularSeasonDetailedResults'], split=True)[0]
weights = st.getSystemWeights(ts, files)
ranks = st.getRanks(ts, files)
elos = st.calcElo(ts_split)

#Add each of them onto the frame to get a fully loaded frame
ts = st.addRanks(ts)
ts = st.addElos(ts)

#Get frames with advanced stats
tsa = st.getStats(ts)

#Get frame with shift stats
ts_shift = st.getInfluenceStats(tsa)

#Merge everything together
tsfull = st.joinFrame(ts, tsa)
tsfull = st.joinFrame(tsfull, ts_shift)

#Get seasonally averaged stats
sts = st.getSeasonalStats(tsfull)

plotter = PlotGenerator(tsfull, sts, files, season=2019)

plotter.showTeamOverall(1140)

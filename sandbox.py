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


files = st.getFiles()
ts = st.getGames(files['MRegularSeasonDetailedResults'])
ts = st.addRanks(ts)
ts = st.addElos(ts)
ts = st.addStats(ts)
ts2019 = ts.loc[ts['Season'] == 2019]
sts = st.getSeasonalStats(ts2019)
wdf = st.getTeamStats(sts, True)
wdf = wdf.drop(columns=['T_Rank', 'T_Elo', 'T_SoS', 'T_Win%', 'T_PythWin%'])
plotter = PlotGenerator(ts, sts, files, season=2019)

kpca = KernelPCA(n_components=5)
pcadf = pd.DataFrame(data=kpca.fit_transform(wdf), index=wdf.index)
aprop = AffinityPropagation(damping=.6)
dbs = DBSCAN(eps=1)
test = aprop.fit(pcadf)

for clust in test.cluster_centers_indices_:
    plotter.showTeamOverall(sts.index.get_level_values(1)[clust])
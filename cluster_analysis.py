#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:05:31 2020

@author: jeff

Cluster Analysis script.

Looking at some of the teams involved in the tournament,
how they compare to the average team.

I wanted a more permanent scripting system for the different mathematical
forays that I'm running.
"""

import numpy as np
import pylab as plab
import pandas as pd
import statslib as st
import seaborn as sns
import matplotlib.pyplot as plt
from plotlib import PlotGenerator, showStat

files = st.getFiles()
ts = st.getGames(files['MRegularSeasonDetailedResults'])
tt = st.getGames(files['MNCAATourneyDetailedResults'])
ts = st.addRanks(ts)
ts = st.addElos(ts)
ts = st.addStats(ts)
ts2019 = ts.loc[ts['Season'] == 2019]
tt2019 = tt.loc[tt['Season'] == 2019].groupby(['Season', 'T_TeamID']).mean()
tte = pd.DataFrame(index=tt2019.index)
sts = st.getSeasonalStats(ts2019)
wdf = st.getTeamStats(sts, True)
wdf = wdf.drop(columns=['T_Rank', 'T_Elo', 'T_SoS', 'T_Win%', 'T_PythWin%'])
stse = tte.merge(wdf, left_index=True, right_index=True)
plotter = PlotGenerator(tt, stse, files, season=2019)

kpca = KernelPCA(n_components=5)
pcadf = pd.DataFrame(data=kpca.fit_transform(wdf), index=wdf.index)
aprop = AffinityPropagation(damping=.6)
dbs = DBSCAN(eps=1)
test = aprop.fit(pcadf)

for clust in test.cluster_centers_indices_:
    plotter.showTeamOverall(sts.index.get_level_values(1)[clust])

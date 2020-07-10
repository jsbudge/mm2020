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
ts = st.addRanks(ts)
ts = st.addElos(ts)
ts = st.addStats(ts)
sts = st.getSeasonalStats(ts)
#wdf = st.getInfluenceStats(ts)
plotter = PlotGenerator(ts, sts, files)
plotter.showTeamOverall(1140)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:07:58 2020

@author: jeff

An ever-expanding statistical analysis of team, game, and seasonal
features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statslib as st
from tqdm import tqdm
from itertools import combinations
import framelib as fl
import featurelib as feat
import seaborn as sns
import eventlib as ev
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures, OneHotEncoder
from scipy.stats import multivariate_normal as mvn
from scipy.stats import iqr
plt.close('all')

files = st.getFiles()
#%%

nspl = 5
scale = StandardScaler()
unscale_df = fl.arrangeFrame(files, scaling=scale, noinfluence=True)
games = fl.arrangeTourneyGames(files, noraw=True)
sdf = unscale_df[0]
score_diff = sdf['T_Score'] - sdf['O_Score']
rank_diff = sdf['T_Rank'] - sdf['O_Rank']
elo_diff = sdf['T_Elo'] - sdf['O_Elo']
avrelo = st.getSeasonalStats(sdf, strat='relelo').drop(columns=['T_Win%', 'T_PythWin%', 'T_SoS'])
#%%

tdf = sdf.drop(columns=[c for c in sdf.columns if c[:1] == 'O'] + ['DayNum'])
tdf['Rank_Diff'] = rank_diff
tdf['Elo_Diff'] = elo_diff

avtdf = avrelo.drop(columns=[c for c in sdf.columns if c[:1] == 'O'])
avtdf['Rank_Diff'] = tdf.groupby(['Season', 'TID']).mean()['Rank_Diff']
avtdf['Elo_Diff'] = tdf.groupby(['Season', 'TID']).mean()['Elo_Diff']

av_cov = tdf.groupby(['Season', 'TID']).apply(lambda x: np.cov(x.T))
    







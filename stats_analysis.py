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
import seaborn as sns
import eventlib as ev
from scipy.stats import multivariate_normal as mvn
from scipy.stats import iqr
plt.close('all')

def getAllAverages(tid):
    ret = pd.DataFrame()
    for av in avs:
        t = avs[av].loc(axis=0)[:, tid].reset_index()
        t['AvType'] = av
        ret = ret.append(t)
    return ret.set_index(['Season', 'TID', 'AvType'])

sdf, sdf_t, sdf_d = st.arrangeFrame(scaling=None, noinfluence=True)
avs = {}
m_types = ['relelo', 'gausselo', 'elo', 'rank', 'mest', 'mean']
for m in tqdm(m_types):
    avs[m] = st.getSeasonalStats(sdf, strat=m)
tdf, tdf_t, tdf_d = st.arrangeTourneyGames()
adv_tdf = st.getTourneyStats(tdf, sdf) 
pdf = ev.getTeamRosters()
tsdf = pd.read_csv('./data/PlayerAnalysisData.csv').set_index(['Season', 'TID'])
print('Scaling for influence...')
inf_df = st.getInfluenceStats(sdf).set_index(['Season', 'TID'])

#%%
print('Differencing...')
diff_df = pd.DataFrame(index=sdf.index, data=sdf[['O_Rank', 'T_Elo', 'O_Elo',
                                                  'T_Score', 'O_Score']])
for col in [col for col in sdf.columns if 'T_' in col]:
    diff_df[col[2:] + 'Diff'] = sdf[col] - sdf['O_' + col[2:]]
diff_avs = {}
for m in tqdm(m_types):
    diff_avs[m] = st.getSeasonalStats(diff_df, strat=m)
    
#%%

#First, how much does any particular stat predict winning?


    







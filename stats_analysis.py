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
from scipy.stats import iqr, chi2_contingency
from sklearn.metrics import mutual_info_score
plt.close('all')

def getPairwiseMI(A, adj=False):
    calc_MI = lambda x, y, bins: \
        mutual_info_score(None, None, 
                          contingency = np.histogram2d(x, y, bins)[0])
    be = {}
    for col in A.columns:
        be[col] = np.histogram_bin_edges(A[col], 'fd')
    matMI = pd.DataFrame(index=A.columns, columns=A.columns)
    for ix in A.columns:
        for jx in A.columns:
            matMI.loc[ix, jx] = calc_MI(A[ix].values, 
                                        A[jx].values, [be[ix], be[jx]])
    if adj:
        kx = np.linalg.pinv(np.sqrt(np.diag(np.diag(matMI.astype(float)))))
        return pd.DataFrame(index=A.columns,
                            columns=A.columns,
                            data=kx.dot(matMI).dot(kx)).astype(float)
    return matMI.astype(float)

def getAllAverages(tid):
    ret = pd.DataFrame()
    for av in avs:
        t = avs[av].loc(axis=0)[:, tid].reset_index()
        t['AvType'] = av
        ret = ret.append(t)
    return ret.set_index(['Season', 'TID', 'AvType'])

sdf, sdf_t, sdf_d = st.arrangeFrame(scaling=None, noinfluence=True)
avs = {}
m_types = ['relelo', 'gausselo', 'elo', 'rank', 'mest', 'mean', 'recent']
for m in tqdm(m_types):
    avs[m] = st.getSeasonalStats(sdf, strat=m)
tdf, tdf_t, tdf_d = st.arrangeTourneyGames()
adv_tdf = st.getTourneyStats(tdf, sdf)
pdf = ev.getTeamRosters()
tsdf = pd.read_csv('./data/PlayerAnalysisData.csv').set_index(['Season', 'TID'])
print('Scaling for influence...')
inf_df = st.getInfluenceStats(sdf).set_index(['Season', 'TID'])

#%%
t_inf = st.getMatches(tdf, inf_df, diff=True)
t_ps = st.getMatches(tdf, tsdf, diff=True)
t_adv = st.getMatches(tdf, adv_tdf.drop(columns=['T_RoundRank']), diff=True)
av_tdf = st.getMatches(tdf, avs['recent'], diff=True)
tdf_diff = st.merge(av_tdf, t_inf, t_ps, t_adv)
tdf_diff['ActualScoreDiff'] = tdf.loc[tdf_diff.index, 'T_Score'] - tdf.loc[tdf_diff.index, 'O_Score']

#%%
#Grouping together using mutual information and correlation metrics
mi_df = getPairwiseMI(tdf_diff, True)
corr_df = abs(tdf_diff.corr())





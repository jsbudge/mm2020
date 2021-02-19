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
from scipy.stats.mstats import gmean, hmean
from scipy.interpolate import CubicSpline
from sklearn.metrics import mutual_info_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.cluster import FeatureAgglomeration
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
import sklearn.linear_model as sk_lm
from itertools import combinations
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

def integerOpt(func, rng, npts=5, nits=5):
    curr_rng = np.linspace(rng[0], rng[1], npts).astype(int)
    curr_vals = np.array([func(i) for i in curr_rng])
    curr_min = np.min(curr_vals)
    m_idx = [n for n in range(npts) if curr_vals[n] == curr_min][0]
    global_min = curr_min + 0.0; global_val = curr_rng[m_idx]
    for it in tqdm(range(nits)):
        if m_idx == 0:
            curr_rng = np.linspace(curr_rng[0], curr_rng[1], npts).astype(int)
        elif m_idx == npts-1:
            curr_rng = np.linspace(curr_rng[-2], curr_rng[-1], npts).astype(int)
        else:
            curr_rng = np.linspace(curr_rng[m_idx-1], curr_rng[m_idx+1], npts).astype(int)
        curr_vals = np.array([func(i) for i in curr_rng])
        curr_min = np.min(curr_vals)
        m_idx = [n for n in range(npts) if curr_vals[n] == curr_min][0]
        if curr_min < global_min:
            global_min = curr_min
            global_val = curr_rng[m_idx]
    return global_val
        
        

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
av_drops = ['T_Rank', 'O_Rank', 'T_Elo', 'O_Elo',
            'T_Win%', 'T_PythWin%', 'T_SoS']
cong_df = st.merge(inf_df, tsdf, avs['mest'],
                   avs['recent'])
tdf_diff = st.getMatches(tdf, cong_df, diff=True)

#%%
ntdiff = tdf_diff.dropna()
ntdiff = ntdiff.drop(columns=['T_RankInf',
                              'T_EloInf', 'DayNumInf'])
#ntdiff = ntdiff.loc[ntdiff.index.get_level_values(0).duplicated(keep='last')]
scores = tdf.loc[ntdiff.index, 'T_Score'] - tdf.loc[ntdiff.index, 'O_Score']
for s, grp in ntdiff.groupby(['Season']):
    ntdiff.loc[grp.index] = (grp - grp.mean()) / grp.std()

#Grouping together using mutual information and correlation metrics
print('Getting Information Stats...')
mi_df = getPairwiseMI(ntdiff, True)
corr_df = abs(ntdiff.corr())
sc_df = np.sqrt(mi_df**2 + corr_df**2) / np.sqrt(2)

def overallScore(n_cl, sc_df, sel_func=np.mean):
    fa = FeatureAgglomeration(n_clusters=n_cl).fit(ntdiff)
    mns = []; l =[]
    for n in range(n_cl):
        if len(ntdiff.columns[fa.labels_ == n]) > 1:
            mns.append(sel_func([sc_df.loc[n] for n in combinations(ntdiff.columns[fa.labels_ == n], 2)]))
            l.append([col for col in ntdiff.columns[fa.labels_ == n]])
    return np.mean(mns), l
    
print('Removing dupe stats...')
#Remove everything that's an exact duplicate
for n in range(20, ntdiff.shape[1] - 1):
    sc, lb = overallScore(n, sc_df, np.min)
    if sc == 1:
        print('{} clusters: {:.2f}'.format(n, sc))
        for l in lb:
            ntdiff = ntdiff.drop(columns=l[1:])
        break
    
    
print('Combining similar stats...')
#Combine the ones that are close to each other
for n in range(20, ntdiff.shape[1] - 1):
    sc, lb = overallScore(n, sc_df, np.min)
    if sc >= .7:
        print('{} clusters: {:.2f}'.format(n, sc))
        for l in lb:
            mcol = l[0]
            mx = abs(np.corrcoef(ntdiff[l[0]], scores)[0, 1])
            for col in l:
                if mx < abs(np.corrcoef(ntdiff[col], scores)[0, 1]):
                    mcol = col
                    mx = abs(np.corrcoef(ntdiff[col], scores)[0, 1])
            ntdiff = ntdiff.drop(columns=[col for col in l if col != mcol])
        break
    
#%%
print('Running regressions...')
Xt, Xs, yt, ys = train_test_split(ntdiff, scores)
rfr = RandomForestRegressor(n_estimators=500)
bayr = sk_lm.BayesianRidge()

#An attempt to create a team-average relative quality stat
rfr.fit(Xt, yt)
bayr.fit(Xt, yt)
res_df = pd.DataFrame()
res_df['RFR'] = rfr.predict(Xs)
res_df['BAY'] = bayr.predict(Xs)
res_df['true'] = ys.values
res_df.index = Xs.index

#%%
#An attempt to get a tournament ranking
allT = pd.DataFrame()
for seas in list(set(ntdiff.index.get_level_values(1))):
    allT = allT.append(st.getAllMatches(cong_df[ntdiff.columns], seas, True))
met_df = pd.DataFrame()

met_df['RFR'] = rfr.predict(allT)
met_df['BAY'] = bayr.predict(allT)
met_df.index = allT.index
mdf_group = met_df.groupby(['Season', 'TID'])
metric = mdf_group.sum() / 68
metric = metric.merge(mdf_group.apply(lambda x: np.sum(x > 0)), on=['Season', 'TID'])
metric = metric.merge(avs['recent'][['T_Rank', 'T_Elo']], on=['Season', 'TID'])
metric = metric.join(adv_tdf[['T_Seed', 'T_RoundRank']], on=['Season', 'TID'])

metric['GeoSCR'] = gmean(metric[['RFR_y', 'BAY_y']].values, axis=1)
metric['MeanSCR'] = metric[['RFR_x', 'BAY_x']].mean(axis=1)

#%%
#Save everything out to a file so we can move between scripts easily
cong_df[ntdiff.columns].to_csv('./data/CongStats.csv')

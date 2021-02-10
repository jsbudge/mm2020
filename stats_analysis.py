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
from scipy.interpolate import CubicSpline
from sklearn.metrics import mutual_info_score
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import KernelPCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from itertools import combinations
from scipy.optimize import minimize
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
t_inf = st.getMatches(sdf, inf_df, diff=True)
t_ps = st.getMatches(sdf, tsdf, diff=True)
t_adv = st.getMatches(tdf, adv_tdf.drop(columns=['T_RoundRank']), diff=True)
av_tdf = st.getMatches(sdf, avs['recent'], diff=True)
tdf_diff = st.merge(av_tdf, t_inf, t_ps)

#%%
ntdiff = tdf_diff.dropna()
ntdiff = ntdiff.loc[ntdiff.index.get_level_values(0).duplicated(keep='last')]
scores = sdf.loc[ntdiff.index, 'T_Score'] - sdf.loc[ntdiff.index, 'O_Score']
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
kdf = ntdiff.astype(np.float32)
n_comp = kdf.shape[1]

tr_s = np.random.permutation(np.arange(kdf.shape[0]))[:int(kdf.shape[0] * .75)]
te_s = [n for n in range(kdf.shape[0]) if n not in tr_s]
yt = scores.values[tr_s]
ys = scores.values[te_s]
Xt = kdf.iloc[tr_s]
Xs = kdf.iloc[te_s]

svr = SVR(kernel='linear')
rfe_select = RFE(svr)
rfe_select.fit(Xt, yt)



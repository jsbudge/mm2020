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
from scipy.stats import iqr, chi2_contingency, rayleigh
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats.mstats import gmean, hmean
from scipy.interpolate import CubicSpline
from sklearn.metrics import mutual_info_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.cluster import FeatureAgglomeration
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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
            if np.isnan(matMI.loc[ix, jx]):
                val = calc_MI(A[ix].values, 
                              A[jx].values, [be[ix], be[jx]])
                matMI.loc[ix, jx] = val
                matMI.loc[jx, ix] = val
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
m_types = ['relelo', 'elo', 'rank', 'mest', 'recent']
for m in tqdm(m_types):
    avs[m] = st.getSeasonalStats(sdf, strat=m)
tdf, tdf_t, tdf_d = st.arrangeTourneyGames()
pdf = ev.getTeamRosters()
tsdf = pd.read_csv('./data/PlayerAnalysisData.csv').set_index(['Season', 'TID'])
print('Scaling for influence...')
inf_df = st.getInfluenceStats(sdf).set_index(['Season', 'TID'])


#%%
#Add in some ideas based on expected values
elodiff = sdf['T_Elo'] - sdf['O_Elo']
expelo = 1 / (1 + 10**(elodiff / 400))
scorediff = sdf['T_Score'] - sdf['O_Score']
eloscoredf = st.merge(pd.DataFrame(elodiff), pd.DataFrame(expelo))

elofit, _ = curve_fit(lambda x, a, b: a * x + b, elodiff, scorediff)

exp_score = scorediff - (elodiff * elofit[0] + elofit[1])
inf_df['ExpScDiff'] = exp_score.groupby(['Season', 'TID']).mean()

#%%
scale = StandardScaler()
cong_df = st.merge(inf_df, tsdf, *[avs[m] for m in m_types]).dropna()
cong_df = pd.DataFrame(index=cong_df.index, columns=cong_df.columns, data=scale.fit_transform(cong_df))
tvsd = TruncatedSVD(n_components=250)

#cong_df = pd.DataFrame(index=cong_df.index, data=scale.fit_transform(tvsd.fit_transform(cong_df)))
tdf_diff = st.getMatches(tdf, cong_df, diff=True)

#%%
ntdiff = tdf_diff.dropna()
#ntdiff = ntdiff.loc[ntdiff.index.get_level_values(0).duplicated(keep='last')]
scores = tdf.loc[ntdiff.index, 'T_Score'] - tdf.loc[ntdiff.index, 'O_Score']

#Grouping together using mutual information and correlation metrics
print('Getting Information Stats...')
mi_df = getPairwiseMI(ntdiff, True)
corr_df = abs(ntdiff.corr())
sc_df = (mi_df + corr_df) / 2

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

#%%
for col in ntdiff.columns:
    if abs(np.corrcoef(ntdiff[col], scores)[1, 0]) < .2:
        if abs(np.corrcoef(ntdiff[col], scores > 0)[1, 0]) < .2:
            cong_df = cong_df.drop(columns=col)
#%%
#Save everything out to a file so we can move between scripts easily
cong_df.to_csv('./data/CongStats.csv')

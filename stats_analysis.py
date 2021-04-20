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
import eventlib as ev
from scipy.optimize import curve_fit
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import RFE
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from itertools import combinations

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
m_types = ['mean', 'relelo', 'recent', 'mest', 'rank', 'elo', 'gausselo', 'median']
for m in tqdm(m_types):
    avs[m] = st.getSeasonalStats(sdf, strat=m)
tdf, tdf_t, tdf_d = st.arrangeTourneyGames()
pdf = ev.getTeamRosters()
tsdf = pd.read_csv('./data/PlayerAnalysisData.csv').set_index(['Season', 'TID'])
print('Scaling for influence...')
inf_df = st.getInfluenceStats(sdf, recalc=False).set_index(['Season', 'TID'])

# Add in some ideas based on expected values
elodiff = sdf['T_Elo'] - sdf['O_Elo']
expelo = 1 / (1 + 10 ** (elodiff / 400))
scorediff = sdf['T_Score'] - sdf['O_Score']
eloscoredf = st.merge(pd.DataFrame(elodiff), pd.DataFrame(expelo))

elofit, _ = curve_fit(lambda x, a, b: a * x + b, elodiff, scorediff)

exp_score = scorediff - (elodiff * elofit[0] + elofit[1])
inf_df['ExpScDiff'] = exp_score.groupby(['Season', 'TID']).mean()

# scale = StandardScaler()
cong_df = st.merge(inf_df, tsdf, *[avs[m] for m in m_types]).dropna()
cong_df = pd.DataFrame(index=cong_df.index, columns=cong_df.columns, data=StandardScaler().fit_transform(cong_df))
tvsd = TruncatedSVD(n_components=cong_df.shape[1] - 1)

# Get all singular values
tvsd.fit(cong_df)

# Find elbow in values, removing noise
min_sv = np.arange(cong_df.shape[1] - 1)[np.log(tvsd.singular_values_) > 2].max()

# Quantile transform to get a more gaussian pdf
cong_df = pd.DataFrame(index=cong_df.index, data=QuantileTransformer().fit_transform(tvsd.transform(cong_df)[:, :min_sv]))
tdf_diff = st.getMatches(tdf, cong_df, diff=True)

ntdiff = tdf_diff.dropna()
# ntdiff = ntdiff.loc[ntdiff.index.get_level_values(0).duplicated(keep='last')]
scores = tdf.loc[ntdiff.index, 'T_Score'] - tdf.loc[ntdiff.index, 'O_Score']

# Grouping together using mutual information and correlation metrics
print('Getting Information Stats...')
sc_df = abs(ntdiff.corr())


def overallScore(n_cl, osc_df, sel_func=np.mean):
    fa = FeatureAgglomeration(n_clusters=n_cl).fit(ntdiff)
    mns = []
    ll = []
    for nn in range(n_cl):
        if len(ntdiff.columns[fa.labels_ == nn]) > 1:
            mns.append(sel_func([osc_df.loc[comb] for comb in combinations(ntdiff.columns[fa.labels_ == nn], 2)]))
            ll.append([col for col in ntdiff.columns[fa.labels_ == nn]])
    return np.mean(mns), ll


print('Removing dupe stats...')
# Remove everything that's an exact duplicate
for n in range(20, ntdiff.shape[1] - 1):
    sc, lb = overallScore(n, sc_df, np.min)
    if sc == 1:
        print('{} clusters: {:.2f}'.format(n, sc))
        for l in lb:
            ntdiff = ntdiff.drop(columns=l[1:])
        break

print('Fitting ABC to remove low scoring columns...')
abc = AdaBoostClassifier(n_estimators=100, learning_rate=1e-4)
rfe = RFE(abc)
rfe.fit(ntdiff, scores)

cong_df = cong_df[cong_df.columns[rfe.support_]]

# Save everything out to a file so we can move between scripts easily
cong_df.to_csv('./data/CongStats.csv')

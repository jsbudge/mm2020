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
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures, OneHotEncoder
import sklearn.cluster as clustering
import sklearn.linear_model as reg
import sklearn.neural_network as nn
import sklearn.decomposition as decomp
from sklearn.pipeline import make_pipeline, FeatureUnion, make_union
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostRegressor
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from scipy.stats import multivariate_normal as mvn
                         

plt.close('all')

files = st.getFiles()
neigs = 5
#final_df = fl.arrangeFrame(files, scaling=PowerTransformer())
unscale_df = fl.arrangeFrame(files, scaling=None, noinfluence=True)
sdf = unscale_df[0]
score_diff = sdf['T_Score'] - sdf['O_Score']
# tdf, twin, _ = fl.arrangeTourneyGames(files)
# ttdf = st.getTourneyStats(tdf, unscale_df[0], files)
avdf = st.getSeasonalStats(sdf, strat='gausselo')
avrelo = st.getSeasonalStats(sdf, strat='relelo')
avrank = st.getSeasonalStats(sdf, strat='rank')
av = st.getSeasonalStats(sdf, strat='mean')
names = st.loadTeamNames(files)

for col in sdf.columns:
    if col[:2] == 'O_':
        sdf = sdf.drop(columns=col)
sdf = sdf.drop(columns=['T_DefRat', 'DayNum'])

U, _lambda, Vt = np.linalg.svd(sdf.drop(columns=['T_Rank', 'T_Elo']), full_matrices=False)

dimred_df = pd.DataFrame(U[:, :neigs], index=sdf.index)
dimred_df = (dimred_df - dimred_df.mean()) / dimred_df.std()
drstddf = dimred_df.groupby(['Season', 'TID']).cov()
dravdf = dimred_df.groupby(['Season', 'TID']).mean()
w = _lambda[:neigs] / sum(_lambda[:neigs])
scale_df = (sdf - sdf.mean()) / sdf.std()

overall = dimred_df.copy()# * w
overall['Perf'] = 0
ovav = overall.groupby(['Season', 'TID']).mean()
relevance = np.corrcoef(scale_df.values.T, score_diff.values)[-1, :-1]
relevance[abs(relevance) < .4] = 0

for idx, grp in dimred_df.groupby(['Season', 'TID']):
    ids = np.logical_and(overall.index.get_level_values(1) == idx[0],
                         overall.index.get_level_values(2) == idx[1])
    try:
        probs = mvn.pdf(dimred_df.loc(axis=0)[:, idx[0], idx[1], :], 
                        dravdf.loc[idx], 
                        drstddf.loc[idx])
        maxprob = mvn.pdf(dravdf.loc[idx], 
                        dravdf.loc[idx], 
                        drstddf.loc[idx])
    except np.linalg.LinAlgError:
        probs = 1
        maxprob = 1
    overall.loc[ids, 'Perf'] = probs / maxprob

stat = 'T_GameScore'
tm = (2011, 1140)
pltdf = pd.DataFrame(sdf.loc(axis=0)[:, tm[0], tm[1], :][stat].values, columns=['x'])
pltdf['y'] = overall.loc(axis=0)[:, tm[0], tm[1], :]['Perf'].values
plt.figure(stat)
plt.scatter(pltdf['x'], pltdf['y'])
plt.plot(av.loc[tm, stat] * np.ones((100,)), np.linspace(0, 1, 100))
plt.plot(np.average(pltdf['x'], 
                    weights=pltdf['y']) * np.ones((100,)),
         np.linspace(0, 1, 100))
sns.kdeplot(pltdf['x'])

plt.figure('Relevance') 
plt.plot(relevance)
nstat = np.sum(((scale_df * (1 / sum(relevance)) / 5) + 1 / sum(relevance)) * relevance, axis=1)

plt.figure('nstat')
sns.kdeplot(nstat.groupby(['Season', 'TID']).mean(), avdf['T_Elo'], levels=40)
#sns.swarmplot(av['T_Score'])
# dr_df = pd.DataFrame(index=unscale_df[2].index).merge(dimred_avdf, right_on=['Season', 'OID'], left_on=['Season', 'TID'],
#                                                       right_index=True).merge(dimred_avdf, right_on=['Season', 'TID'], left_on=['Season', 'OID'],
#                                                       right_index=True).sort_index()
# dr_df = pd.DataFrame(index=sdf.index).merge(dimred_df, right_on=['Season', 'OID'], left_on=['Season', 'TID'],
#                                                       right_index=True).sort_index()
# gav_df = pd.DataFrame(index=sdf.index).merge(avdf, right_on=['Season', 'OID'], left_on=['Season', 'TID'],
#                                                       right_index=True).merge(avdf, right_on=['Season', 'TID'], left_on=['Season', 'OID'],
#                                                       right_index=True).sort_index()
                                                                              
# wins = pd.DataFrame(unscale_df[1].values, columns=['Win'], index=unscale_df[2].index).sort_index()
# g1_df = dr_df.loc[dr_df.index.get_level_values(0).duplicated(keep='first')]
# g2_df = dr_df.loc[dr_df.index.get_level_values(0).duplicated(keep='last')]
#pltdf = sdf[['T_Rank', ''
#sns.pairplot(


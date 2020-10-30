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
neigs = 17
#final_df = fl.arrangeFrame(files, scaling=PowerTransformer())
unscale_df = fl.arrangeFrame(files, scaling=StandardScaler())
sdf = unscale_df[0]
# tdf, twin, _ = fl.arrangeTourneyGames(files)
# ttdf = st.getTourneyStats(tdf, unscale_df[0], files)
avdf = st.getSeasonalStats(sdf, strat='gausselo')
#names = st.loadTeamNames(files)

for col in sdf.columns:
    if col[:2] == 'O_':
        sdf = sdf.drop(columns=col)
    elif col[-5:] == 'Shift':
        sdf = sdf.drop(columns=col)

U, _lambda, Vt = np.linalg.svd(sdf.drop(columns=['T_Rank', 'T_Elo']), full_matrices=False)

dimred_df = pd.DataFrame(U[:, :neigs], index=sdf.index)
dimred_df = (dimred_df - dimred_df.mean()) / dimred_df.std()
drstddf = dimred_df.groupby(['Season', 'TID']).cov()
w = _lambda[:neigs] / sum(_lambda[:neigs])

overall = np.sum(dimred_df * w, axis=1)
ovav = overall.groupby(['Season', 'TID']).mean()
ovstd = overall.groupby(['Season', 'TID']).std()

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
pltdf = sdf[['T_Rank', ''
sns.pairplot(


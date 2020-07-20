#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:05:31 2020

@author: jeff

Cluster Analysis script.

Looking at some of the teams involved in the tournament,
how they compare to the average team.

I wanted a more permanent scripting system for the different mathematical
forays that I'm running.
"""

import numpy as np
import pylab as plab
import pandas as pd
import statslib as st
import seaborn as sns
import matplotlib.pyplot as plt
from plotlib import PlotGenerator, showStat
from tourney import FeatureCreator
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, \
    PolynomialFeatures
from sklearn.cluster import AffinityPropagation, DBSCAN, KMeans, MeanShift, SpectralClustering, FeatureAgglomeration
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import KernelPCA, FastICA, LatentDirichletAllocation, DictionaryLearning
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif, f_classif, \
    SelectPercentile, VarianceThreshold

files = st.getFiles()
fc = FeatureCreator(files, scaling=RobustScaler())
# Xtrans, ytrans = fc.splitGames(st.getGames(files['MNCAATourneyDetailedResults']))
# transform = FeatureUnion([('kpca_lin', KernelPCA(n_components=20, kernel='linear')),
#                           ('kpca_rbf', KernelPCA(n_components=20, kernel='rbf')),
#                           ('scores', Pipeline([('poly', PolynomialFeatures(degree=2)),
#                                                ('variance', VarianceThreshold()),
#                                                ('minfo', SelectPercentile(score_func=mutual_info_classif, percentile=10)),
#                                                ('fclass', SelectPercentile(score_func=f_classif, percentile=10))]))])
# fc.reTransform(transform.fit(Xtrans, ytrans))

#%%
test = FeatureAgglomeration(n_clusters=None, compute_full_tree=True, linkage='average', distance_threshold=10).fit(fc.avdf)

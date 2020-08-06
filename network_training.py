#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 16:14:31 2020

@author: jeff

Neural Net Model
"""

import numpy as np
import pylab as plab
import pandas as pd
import statslib as st
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import HuberRegressor, Lars, ElasticNet, Lasso, SGDRegressor, TheilSenRegressor, \
    ARDRegression, LassoLars
from sklearn.decomposition import KernelPCA, FastICA, LatentDirichletAllocation, DictionaryLearning, \
    TruncatedSVD
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif, f_classif, \
    SelectPercentile, VarianceThreshold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVR, SVC
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    RandomForestRegressor
from sklearn.manifold import Isomap
from sklearn.covariance import EmpiricalCovariance, OAS
from sklearn.metrics import log_loss, make_scorer
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer, \
    PolynomialFeatures, MinMaxScaler
from plotlib import showStat, showCorrs
from scipy.optimize import minimize
from tourney import Bracket, FeatureFrame, GameFrame
import framelib as fl
import featurelib as feat
from tabulate import tabulate
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers

#Gather all of our files
# split_yr = 2018
files = st.getFiles()
cv = feat.seasonalCV(np.arange(2004, 2020))
trans = None
finfo = SelectPercentile(score_func=f_classif, percentile=25)

print('Loading features...')
fc = fl.arrangeFrame(files, scaling=None, noraw=True)
fc[0]['T_TOffRat'] = feat.proprietaryFeats(fc[0])
ftc = fl.arrangeTourneyGames(files, noraw=True)
ts = st.getTourneyStats(ftc[0], fc[0], files)
sts = fl.merge(st.getSeasonalStats(fc[0], strat='elo'), 
               ts[['T_Seed', 'T_FinalElo', 'T_FinalRank']])

#%%
nodes = 650
gc, gc_y = fl.loadGames(sts, ftc[0])
pipe = make_pipeline(VarianceThreshold(),
                    StandardScaler()).fit(gc, gc_y)
gck = pipe.transform(gc)

Xt, Xs, yt, ys = train_test_split(gck, gc_y.values, test_size=.25)
onehot = OneHotEncoder(sparse=False)
yt = onehot.fit_transform(yt.reshape(-1, 1))
ys = onehot.fit_transform(ys.reshape(-1, 1))

model = Sequential([
    layers.Input(shape=(Xt.shape[1],)),
    layers.Dense(nodes, activation='relu'),
    layers.GaussianNoise(2),
    layers.Dense(nodes, activation='relu'),
    layers.Dropout(.25),
    layers.Dense(nodes, activation='relu'),
    layers.Dense(nodes, activation='relu'),
    layers.Dropout(.25),
    layers.Dense(nodes, activation='relu'),
    layers.Dense(nodes, activation='relu'),
    layers.Dropout(.25),
    layers.Dense(nodes, activation='relu'),
    layers.ActivityRegularization(),
    layers.Dense(nodes, activation='relu'),
    layers.Dropout(.25),
    layers.Dense(nodes, activation='relu'),
    layers.Dense(nodes, activation='relu'),
    layers.Dropout(.25),
    layers.Dense(nodes, activation='relu'),
    layers.Dense(nodes, activation='relu'),
    layers.Dropout(.25),
    layers.Dense(nodes, activation='relu'),
    layers.Dense(nodes, activation='relu'),
    layers.Dense(2, activation='softmax')
    ])

optimizer = keras.optimizers.Adam(learning_rate=.0001, amsgrad=True)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(Xt, yt, validation_data=(Xs, ys), epochs=50)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

bracket = Bracket(2019, files, sts)
bracket.run(model, pipe)
bracket.printTree('./test.txt')

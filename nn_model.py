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
import eventlib as ev
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures, OneHotEncoder
import sklearn.cluster as clustering
import sklearn.linear_model as reg
import sklearn.neural_network as nn
import sklearn.decomposition as decomp
from sklearn.pipeline import make_pipeline, FeatureUnion, make_union
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostRegressor
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from scipy.stats import multivariate_normal as mvn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
plt.close('all')

def compile_model(state_sz, n_layers=5, layer_sz=50, optimizer='adam'):
    inp1 = keras.Input(shape=(state_sz,))
    inp2 = keras.Input(shape=(state_sz,))
    mdlconc = layers.Subtract()([inp1, inp2])
    out = layers.Dense(layer_sz)(mdlconc)
    for n in range(n_layers):
        out = layers.Dense(layer_sz)(out)
        out = layers.Dropout(.1)(out)
    smax = layers.Dense(2, activation='softmax')(out)
    model = keras.Model(inputs=[inp1, inp2], outputs=smax)
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'],
                  metrics=['accuracy'])
    return model

files = st.getFiles()
#%%

nspl = 5
scale = StandardScaler()
cv = KFold(n_splits=nspl, shuffle=True)
unscale_df = fl.arrangeFrame(files, scaling=scale, noinfluence=True)
games = fl.arrangeTourneyGames(files, noraw=True)
sdf = unscale_df[0]
score_diff = sdf['T_Score'] - sdf['O_Score']
avdf = st.getSeasonalStats(sdf, strat='gausselo').drop(columns=['T_PythWin%', 'T_SoS'])
avrelo = st.getSeasonalStats(sdf, strat='relelo').drop(columns=['T_PythWin%', 'T_SoS'])
avrank = st.getSeasonalStats(sdf, strat='rank').drop(columns=['T_PythWin%', 'T_SoS'])
av = st.getSeasonalStats(sdf, strat='mean').drop(columns=['T_PythWin%', 'T_SoS'])
#%%


outcomes = OneHotEncoder(sparse=False).fit_transform(pd.DataFrame(data=games[1].values, index=games[0].index).sort_index().values + 0)

callbacks = [tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    min_delta=1e-3,
                    patience=60,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=True),
                 tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss",
                    factor=0.5,
                    patience=20,
                    verbose=0,
                    mode="auto",
                    min_delta=0.0001,
                    cooldown=0,
                    min_lr=1e-10)]

for frame in [('gausselo', avdf), ('relelo', avrelo),
              ('rank', avrank), ('mean', av)]:
    print(frame[0])
    g1 = pd.DataFrame(index=games[0].index).sort_index().reset_index().merge(frame[1], on=['Season', 'TID'], how='left')
    g1 = g1.set_index(['GameID', 'TID']).drop(columns=['OID', 'Season'])
    g2 = pd.DataFrame(index=games[0].index).sort_index().reset_index().merge(frame[1], left_on=['Season', 'OID'], right_on=['Season', 'TID'], how='left')
    g2 = g2.set_index(['GameID', 'OID']).drop(columns=['TID', 'Season'])
    av_loss = np.zeros((nspl,)); av_acc = np.zeros((nspl,))
    for n, (tt, ts) in enumerate(cv.split(g1)):
        model = compile_model(g1.shape[1], n_layers=20, layer_sz=100, optimizer='adam')
        Xt1, Xs1 = g1.iloc[tt], g1.iloc[ts]
        Xt2, Xs2 = g2.iloc[tt], g2.iloc[ts]
        yt, ys = outcomes[tt], outcomes[ts]
        h = model.fit([Xt1, Xt2], yt, epochs=450, validation_data=([Xs1, Xs2], ys),
                  callbacks=callbacks, verbose=0)
        mdl_loss, mdl_acc = model.evaluate([Xs1, Xs2], ys, verbose=0)
        av_loss[n] += mdl_loss / nspl
        av_acc[n] += mdl_acc / nspl
        print('\tFold {} complete.'.format(n))
    







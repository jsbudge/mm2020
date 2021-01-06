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
from sklearn.feature_selection import f_regression, mutual_info_classif, SelectPercentile
from scipy.stats import multivariate_normal as mvn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
from keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
from datetime import datetime
plt.close('all')

'''
Compile keras model using the parameters listed.
state_sz: the size of each feature vector to be used in training/prediction.
optimizer: the optimization method for the model. Adam is the default,
            but there are plenty of others.
'''
def compile_model(state_sz, optimizer='adam'):
    inp1 = keras.Input(shape=(state_sz,))
    inp2 = keras.Input(shape=(state_sz,))
    mdlconc = layers.Subtract()([inp1, inp2])
    out = layers.Dense(60, activation='relu')(mdlconc)
    out = layers.Dense(50, activation='relu')(out)
    out = layers.Dense(50, activation='relu')(out)
    out = layers.Dropout(.3)(out)
    out = layers.Dense(50, activation='relu')(out)
    out = layers.Dense(50, activation='relu')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(50, activation='relu')(out)
    out = layers.Dense(50, activation='relu')(out)
    out = layers.Dense(50, activation='relu')(out)
    out = layers.Dropout(.3)(out)
    out = layers.Dense(50, activation='relu')(out)
    out = layers.Dense(50, activation='relu')(out)
    out = layers.Dense(50, activation='relu')(out)
    out = layers.Dropout(.3)(out)
    out = layers.Dense(50, activation='relu')(out)
    out = layers.Dense(50, activation='relu')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Dense(50, activation='relu')(out)
    out = layers.Dense(50, activation='relu')(out)
    out = layers.Dense(50, activation='relu')(out)
    out = layers.Dropout(.3)(out)
    out = layers.Dense(50, activation='relu')(out)
    smax = layers.Dense(2, activation='softmax',
                        name='output')(out)
    model = keras.Model(inputs=[inp1, inp2], outputs=smax)
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'],
                  metrics=['accuracy'])
    return model

'''
Gets the frames needed for training, plus the outcomes of the games.
idx: DataFrame Index which holds the names of teams for games.
df: DataFrame with the feature vectors that describe each team.
targets: Series with True or False depending on if TID won the game.
'''
def get_frames(idx, df, targets):
    g1 = pd.DataFrame(index=idx).sort_index().reset_index().merge(df, on=['Season', 'TID'], how='left')
    g1 = g1.set_index(['GameID', 'TID']).drop(columns=['OID', 'Season'])
    g2 = pd.DataFrame(index=idx).sort_index().reset_index().merge(df, left_on=['Season', 'OID'], right_on=['Season', 'TID'], how='left')
    g2 = g2.set_index(['GameID', 'OID']).drop(columns=['TID', 'Season'])
    outcomes = OneHotEncoder(sparse=False).fit_transform(pd.DataFrame(data=targets.values, index=idx).sort_index().values + 0)
    return g1, g2, outcomes

files = st.getFiles()

#%%

nspl = 5
scale = StandardScaler()
cv = KFold(n_splits=nspl, shuffle=True)
scale_df = fl.arrangeFrame(files, scaling=StandardScaler(), noinfluence=True)
unscale_df = fl.arrangeFrame(files, scaling=None, noinfluence=True)
games = fl.arrangeTourneyGames(files, noraw=True)
sdf = scale_df[0]
t_df = st.getTourneyStats(games[0], unscale_df[0], files)
score_diff = unscale_df[0]['T_Score'] - unscale_df[0]['O_Score']
seasonal_stats = st.getSeasonalStats(unscale_df[0], seasonal_only=True)
for season, df in seasonal_stats.groupby('Season'):
        seasonal_stats.loc(axis=0)[season, :] = scale.fit_transform(df)
avdf = st.getSeasonalStats(sdf, strat='relelo').drop(columns=['T_PythWin%', 'T_Win%', 'T_SoS']).merge(seasonal_stats, on=['Season', 'TID'])

#Benchmarks to test our net against
e1, e2, e_out = get_frames(games[0].index, t_df, games[1])
elo_bench = sum(np.logical_and(e1['T_FinalElo'].values - e2['T_FinalElo'].values < 0, e_out[:, 0])) / e1.shape[0] + \
    sum(np.logical_and(e1['T_FinalElo'].values - e2['T_FinalElo'].values > 0, e_out[:, 1])) / e1.shape[0]
rank_bench = sum(np.logical_and(e1['T_FinalRank'].values - e2['T_FinalRank'].values > 0, e_out[:, 0])) / e1.shape[0] + \
    sum(np.logical_and(e1['T_FinalRank'].values - e2['T_FinalRank'].values < 0, e_out[:, 1])) / e1.shape[0]
eq_sds = e1['T_Seed'].values - e2['T_Seed'].values != 0
seed_bench = sum(np.logical_and(e1['T_Seed'].values[eq_sds] - e2['T_Seed'].values[eq_sds] > 0, e_out[eq_sds, 0])) / sum(eq_sds) + \
    sum(np.logical_and(e1['T_Seed'].values[eq_sds] - e2['T_Seed'].values[eq_sds] < 0, e_out[eq_sds, 1])) / sum(eq_sds)
    
#%%

#Set up probabilistic forecast, teach it to know what wins games
prob_df = sdf.drop(columns=[c for c in sdf.columns if c[:1] == 'O'] + \
                   ['DayNum', 'T_Score', 'T_FGM', 'T_FGA', 'T_Poss', 'T_OffRat', 'T_DefRat'])
pdf = prob_df.groupby(['GameID']).first() - prob_df.groupby(['GameID']).last()
pdf = pdf.append(-pdf).sort_index().set_index(prob_df.sort_index().index)

kpca = decomp.TruncatedSVD(n_components=25)
kpca.fit(pdf)
ff = pd.DataFrame(index=pdf.index, data=kpca.transform(pdf)).groupby(['Season', 'TID']).mean()
# plt.figure('KPCA Eigenvalues')
# plt.subplot(211)
# plt.plot(kpca.lambdas_)
# plt.subplot(212)
# sns.heatmap(ff.cov())

g1, g2, outcomes = get_frames(sdf.index, ff, scale_df[1])
e1, e2, e_out = get_frames(games[0].index, ff, games[1])

#%%
K.clear_session()

learn_rate = 1e-4
num_epochs = 5000
callbacks = [tf.keras.callbacks.EarlyStopping(
                    monitor="loss",
                    min_delta=1e-4,
                    patience=200,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=True),
            TensorBoard(histogram_freq=3, write_images=False,
                        log_dir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S"))]

opt_adam = keras.optimizers.Adam(learning_rate=learn_rate)
model = compile_model(g1.shape[1], optimizer=opt_adam)

t_pts, v_pts = next(cv.split(outcomes))
Xt1, Xs1 = g1.iloc[t_pts], g1.iloc[v_pts]
Xt2, Xs2 = g2.iloc[t_pts], g2.iloc[v_pts]
yt, ys = outcomes[t_pts], outcomes[v_pts]
hist = model.fit([Xt1, Xt2], yt, epochs=num_epochs, validation_data=([Xs1, Xs2], ys),
          callbacks=callbacks, verbose=2, batch_size=400)
model.evaluate([e1, e2], e_out)







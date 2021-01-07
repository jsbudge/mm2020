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
from tensorboard.plugins.hparams import api as hp
from datetime import datetime
plt.close('all')

'''
Compile keras model using the parameters listed.
state_sz: the size of each feature vector to be used in training/prediction.
optimizer: the optimization method for the model. Adam is the default,
            but there are plenty of others.
'''
def compile_model(state_sz, layer_sz, n_layers, optimizer='adam', metrics=['accuracy']):
    dropout_rate = .2
    inp1 = keras.Input(shape=(state_sz,))
    inp2 = keras.Input(shape=(state_sz,))
    mdlconc = layers.Subtract()([inp1, inp2])
    out = layers.Dense(layer_sz, activation='relu', name='init_dense')(mdlconc)
    for n in range(n_layers):
        out = layers.Dense(layer_sz, activation='relu',
                           name='dense_{}_0'.format(n))(out)
        out = layers.Dense(layer_sz, activation='relu',
                           name='dense_{}_1'.format(n))(out)
        out = layers.Dropout(dropout_rate)(out)
        out = layers.Dense(layer_sz, activation='relu',
                           name='dense_{}_2'.format(n))(out)
        out = layers.Dense(layer_sz, activation='relu',
                           name='dense_{}_3'.format(n))(out)
        out = layers.BatchNormalization(name='bn_{}'.format(n))(out)
        out = layers.Dense(layer_sz, activation='relu',
                           name='dense_{}_4'.format(n))(out)
        out = layers.Dense(layer_sz, activation='relu',
                           name='dense_{}_5'.format(n))(out)
        out = layers.Dense(layer_sz, activation='relu',
                           name='dense_{}_6'.format(n))(out)
        out = layers.Dropout(dropout_rate)(out)
    out = layers.Dense(layer_sz, activation='relu',
                       name='final_dense')(out)
    smax = layers.Dense(2, activation='softmax',
                        name='output')(out)
    model = keras.Model(inputs=[inp1, inp2], outputs=smax)
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'],
                  metrics=metrics)
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
tune_hyperparams = False
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

kpca = decomp.TruncatedSVD(10)
kpca.fit(pdf)
ff = pd.DataFrame(index=pdf.index, data=kpca.transform(pdf)).groupby(['Season', 'TID']).mean()

cmb_df = avdf.merge(ff, left_index=True, right_index=True)

g1, g2, outcomes = get_frames(sdf.index, cmb_df, scale_df[1])
e1, e2, e_out = get_frames(games[0].index, cmb_df, games[1])

#%%
K.clear_session()

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([20, 40, 60, 100]))
HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([1, 2, 3, 4]))
HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-2, 1e-3, 1e-4]))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
      hparams=[HP_NUM_UNITS, HP_NUM_LAYERS, HP_LR],
      metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

learn_rate = 1e-3
num_epochs = 300
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
k_calls = [tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-4,
                    patience=50,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=True),
            TensorBoard(histogram_freq=3, write_images=False,
                        log_dir=logdir)]
metrics = ['accuracy',
           tf.keras.metrics.AUC(),
           tf.keras.metrics.Precision(),
           tf.keras.metrics.Recall()]

opt_adam = keras.optimizers.Adam(learning_rate=learn_rate)


t_pts, v_pts = next(cv.split(outcomes))
Xt1, Xs1 = g1.iloc[t_pts], g1.iloc[v_pts]
Xt2, Xs2 = g2.iloc[t_pts], g2.iloc[v_pts]
yt, ys = outcomes[t_pts], outcomes[v_pts]

if tune_hyperparams:
    session_num = 0
    for num_units in HP_NUM_UNITS.domain.values:
        for num_layers in HP_NUM_LAYERS.domain.values:
            for lrate in HP_LR.domain.values:
                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_NUM_LAYERS: num_layers,
                    HP_LR: lrate,
                }
                callbacks = k_calls + [hp.KerasCallback(logdir, hparams)]
                model = compile_model(g1.shape[1], hparams[HP_NUM_UNITS], 
                                      hparams[HP_NUM_LAYERS], 
                                      optimizer = keras.optimizers.Adam(learning_rate=hparams[HP_LR]),
                                      metrics=metrics)
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                with tf.summary.create_file_writer('logs/hparam_tuning/' + run_name).as_default():
                    hp.hparams(hparams)  # record the values used in this trial
                    model.fit([Xt1, Xt2], yt, epochs=num_epochs, validation_data=([Xs1, Xs2], ys),
                              callbacks=callbacks, verbose=2, batch_size=200)
                    mets = model.evaluate([Xs1, Xs2], ys)
                    tf.summary.scalar(METRIC_ACCURACY, mets[1], step=1)
                session_num += 1
else:
    model = compile_model(g1.shape[1], 80, 
                        1, 
                        optimizer = keras.optimizers.Adam(learning_rate=1e-4),
                        metrics=metrics)
    model.fit([Xt1, Xt2], yt, epochs=num_epochs, validation_data=([Xs1, Xs2], ys),
              callbacks=k_calls, verbose=2, batch_size=200)
model.evaluate([e1, e2], e_out)






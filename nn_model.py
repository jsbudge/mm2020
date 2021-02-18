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
from tensorflow.keras import layers, regularizers
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
    dropout_rate = .4
    inp = keras.Input(shape=(state_sz[1],))
    out = layers.Dense(layer_sz, activation='relu', name='conc_dense',
                           kernel_regularizer=regularizers.l2(1e-1),
                           bias_regularizer=regularizers.l2(1e-1))(inp)
    out = layers.Flatten()(out)
    out = layers.Dense(layer_sz, activation='relu', name='init_dense',
                           kernel_regularizer=regularizers.l2(1e-1),
                           bias_regularizer=regularizers.l2(1e-1))(out)
    out = layers.Dropout(dropout_rate)(out)
    for n in range(n_layers):
        out = layers.Dense(layer_sz, activation='relu',
                           name='dense_{}'.format(n),
                           kernel_regularizer=regularizers.l2(1e-1),
                           bias_regularizer=regularizers.l2(1e-1))(out)
        out = layers.Dropout(dropout_rate)(out)
    out = layers.Dense(layer_sz, activation='relu',
                       name='final_dense')(out)
    smax = layers.Dense(2, activation='softmax',
                        name='output')(out)
    model = keras.Model(inputs=inp, outputs=smax)
    model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy'],
                  metrics=metrics)
    return model

def rescale(df, scaler):
    return pd.DataFrame(index=df.index, columns=df.columns,
                        data=scaler.transform(df))

#%%

tune_hyperparams = True
scale = StandardScaler()

sdf, sdf_t, sdf_d = st.arrangeFrame(scaling=None, noinfluence=True)
tdf, tdf_t, tdf_d = st.arrangeTourneyGames()
adv_tdf = st.getTourneyStats(tdf, sdf)
st_df = pd.read_csv('./data/CongStats.csv').set_index(['Season', 'TID'])

st_df = st_df.merge(adv_tdf.drop(columns=['T_RoundRank']), on=['Season', 'TID'])
ml_df = st.getMatches(tdf, st_df, diff=True)
target = OneHotEncoder(sparse=False).fit_transform(tdf_t[ml_df.index].values.reshape((-1, 1)))

#Benchmarks to test our net against
elo_bench = sum(np.logical_and(ml_df['T_FinalElo'].values < 0, target[:, 0])) / ml_df.shape[0] + \
    sum(np.logical_and(ml_df['T_FinalElo'].values > 0, target[:, 1])) / ml_df.shape[0]
rank_bench = sum(np.logical_and(ml_df['T_FinalRank'].values > 0, target[:, 0])) / ml_df.shape[0] + \
    sum(np.logical_and(ml_df['T_FinalRank'].values < 0, target[:, 1])) / ml_df.shape[0]
#%%

#Split features into a train and test set
Xt, Xs, yt, ys = train_test_split(ml_df, target)
scale.fit(Xt)
Xt = rescale(Xt, scale)
Xs = rescale(Xs, scale)

#Do some imputing here to weight towards champs, hopefully


#%%
K.clear_session()

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([100, 200, 300, 400, 500, 600, 700, 800, 900]))
HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([1, 2, 3, 4, 5]))
HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-3, 1e-4, 1e-5, 1e-6]))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
      hparams=[HP_NUM_UNITS, HP_NUM_LAYERS, HP_LR],
      metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

learn_rate = 1e-3
num_epochs = 800
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
k_calls = [tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-4,
                    patience=20,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=True),
            TensorBoard(histogram_freq=3, write_images=False,
                        log_dir=logdir)]
metrics = ['accuracy',
           tf.keras.metrics.AUC()]

opt_adam = keras.optimizers.Adam(learning_rate=learn_rate)

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
                model = compile_model(Xt.shape, hparams[HP_NUM_UNITS], 
                                      hparams[HP_NUM_LAYERS], 
                                      optimizer = keras.optimizers.Adam(learning_rate=hparams[HP_LR]),
                                      metrics=metrics)
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                with tf.summary.create_file_writer('logs/hparam_tuning/' + run_name).as_default():
                    hp.hparams(hparams)  # record the values used in this trial
                    model.fit(Xt, yt, epochs=num_epochs, validation_data=(Xs, ys),
                              callbacks=callbacks, verbose=2, batch_size=64)
                    mets = model.evaluate(Xs, ys)
                    tf.summary.scalar(METRIC_ACCURACY, mets[1], step=1)
                session_num += 1
else:
    model = compile_model(Xt.shape, 100, 
                        3, 
                        optimizer = keras.optimizers.Adam(learning_rate=1e-5),
                        metrics=metrics)
    model.fit(Xt, yt, epochs=num_epochs, validation_data=(Xs, ys),
              callbacks=k_calls, verbose=2, batch_size=64)






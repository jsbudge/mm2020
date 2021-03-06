#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:07:58 2020

@author: jeff

An ever-expanding statistical analysis of team, game, and seasonal
features.
"""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers, regularizers

import statslib as st
from tourney import Bracket

plt.close('all')

'''
Compile keras model using the parameters listed.
state_sz: the size of each feature vector to be used in training/prediction.
optimizer: the optimization method for the model. Adam is the default,
            but there are plenty of others.
'''


def compile_model(state_sz, layer_sz, n_layers, optimizer='adam', metrics=['accuracy']):
    dropout_rate = .5
    inp = keras.Input(shape=(state_sz[1],))
    out = layers.Dense(layer_sz, activation='relu', name='init_dense',
                       kernel_regularizer=regularizers.l2(1e-1),
                       bias_regularizer=regularizers.l2(1e-1))(inp)
    if n_layers > 1:
        out = layers.Dropout(dropout_rate)(out)
        for n in range(n_layers - 1):
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


def shuffle(df):
    idx = np.random.permutation(np.arange(df.shape[0]))
    return df.iloc[idx], idx


# %%
print('Loading raw data...')
tune_hyperparams = False
scale = StandardScaler()
scale_st2 = StandardScaler()
names = st.loadTeamNames()

sdf, sdf_t, sdf_d = st.arrangeFrame(scaling=None, noinfluence=True)
sdf_t.index = sdf.index
tdf, tdf_t, tdf_d = st.arrangeTourneyGames()
adv_tdf = st.getTourneyStats(tdf, sdf)
st_df = pd.read_csv('./data/CongStats.csv').set_index(['Season', 'TID'])

ml_df = st.getMatches(sdf, st_df, diff=True).astype(np.float32).dropna()
target = OneHotEncoder(sparse=False).fit_transform(sdf_t[ml_df.index].values.reshape((-1, 1)))
sdf = sdf.loc[ml_df.index]

# Benchmarks to test our net against
elo_bench = sum(np.logical_and(sdf['T_Elo'].values - sdf['O_Elo'].values < 0,
                               target[:, 0])) / ml_df.shape[0] + \
            sum(np.logical_and(sdf['T_Elo'].values - sdf['O_Elo'].values > 0,
                               target[:, 1])) / ml_df.shape[0]
rank_bench = sum(np.logical_and(sdf['T_Rank'].values - sdf['O_Rank'].values > 0,
                                target[:, 0])) / ml_df.shape[0] + \
             sum(np.logical_and(sdf['T_Rank'].values - sdf['O_Rank'].values < 0,
                                target[:, 1])) / ml_df.shape[0]
# %%
print('Splitting and rescaling...')
# Split features into a train and test set
Xt, Xs, yt, ys = train_test_split(ml_df, target)
scale.fit(Xt)
Xt = rescale(Xt, scale)
Xs = rescale(Xs, scale)

# %%
K.clear_session()

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([200, 500, 800, 1000]))
HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([1, 2]))
HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-3, 1e-4, 5e-5, 1e-5]))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_NUM_LAYERS, HP_LR],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
    )

learn_rate = 1e-4
num_epochs = 6000
n_layers = 10
n_nodes = 150
runID = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/fit/" + runID
k_calls = [tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=1e-5,
    patience=40,
    verbose=2,
    mode="auto",
    baseline=None,
    restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=2,
        mode="auto",
        min_delta=1e-4,
        cooldown=3,
        min_lr=1e-8),
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
                                      optimizer=keras.optimizers.Adam(learning_rate=hparams[HP_LR]),
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
    model = compile_model(Xt.shape, n_nodes, n_layers,
                          optimizer=opt_adam,
                          metrics=metrics)
    model.fit(Xt, yt, epochs=num_epochs, validation_data=(Xs, ys),
              callbacks=k_calls, verbose=2)

# %%
print('Splitting and rescaling stage 2...')
# Training sets for tournament data
tml_df = st.getMatches(tdf, st_df, diff=True).astype(np.float32).dropna()
tml_df = rescale(tml_df, scale)

tdiff_df = st.getMatches(tdf, adv_tdf, diff=True).astype(np.float32).dropna()

# Get splits for validation data of a single tournament
Xt_t, _ = shuffle(tml_df.loc(axis=0)[:, :2017, :, :])
yt_t = OneHotEncoder(sparse=False).fit_transform(tdf_t[Xt_t.index].values.reshape((-1, 1)))
Xs_t, _ = shuffle(tml_df.loc(axis=0)[:, 2018:, :, :])
ys_t = OneHotEncoder(sparse=False).fit_transform(tdf_t[Xs_t.index].values.reshape((-1, 1)))

# %%

res_df = pd.DataFrame(index=Xs_t.index)
tids = res_df.index.get_level_values(2)
oids = res_df.index.get_level_values(3)
scores = tdf.loc[Xs_t.index, 'T_Score'] - tdf.loc[Xs_t.index, 'O_Score']
base_preds = model.predict(Xs_t)
res_df['T_Win%'] = base_preds[:, 0]
res_df['O_Win%'] = base_preds[:, 1]
res_df['PredWin'] = [tids[n] if base_preds[n, 0] > .5 else oids[n] for n in range(Xs_t.shape[0])]
res_df['TrueWin'] = [tids[n] if scores.iloc[n] > 0 else oids[n] for n in range(Xs_t.shape[0])]
res_df['logloss'] = -(ys_t[:, 0] * np.log(base_preds[:, 0]) + ys_t[:, 1] * np.log(base_preds[:, 1]))
base_acc = sum(res_df['PredWin'] == res_df['TrueWin']) / res_df.shape[0]

# %%

Xs2_t = tdiff_df.loc[Xs_t.index]
Xt2_t = tdiff_df.loc[Xt_t.index]

scale_st2.fit(Xt2_t)
Xt2_t = rescale(Xt2_t, scale_st2)
Xs2_t = rescale(Xs2_t, scale_st2)

# %%

# Use the old model for some good ol' transfer learning, adding seeding stats and
# other stuff that is tournament-specific
# Freeze the layers in the first level model
model.trainable = False
n_augnodes = tdiff_df.shape[1]

m_inp = keras.Input(shape=(n_augnodes,))
mx = layers.Dense(n_augnodes, activation='relu', name='dense_aug1',
                  kernel_regularizer=regularizers.l2(1e-1),
                  bias_regularizer=regularizers.l2(1e-1))(m_inp)
mx = layers.Dense(n_augnodes, activation='relu', name='dense_aug2',
                  kernel_regularizer=regularizers.l2(1e-1),
                  bias_regularizer=regularizers.l2(1e-1))(mx)

mx = layers.Concatenate()([model.layers[-1].output, mx])
mx = layers.Dense(n_nodes, activation='relu', name='dense_augconc',
                  kernel_regularizer=regularizers.l2(1e-1),
                  bias_regularizer=regularizers.l2(1e-1))(mx)
mx = layers.Dense(2, activation='softmax', name='aug_output')(mx)

m2 = keras.Model(inputs=[model.input, m_inp], outputs=mx)
m2.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
           loss=['binary_crossentropy'],
           metrics=metrics)

k_calls = [tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=40,
    verbose=2,
    mode="auto",
    baseline=None,
    restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=10,
        verbose=2,
        mode="auto",
        min_delta=1e-4,
        cooldown=0,
        min_lr=1e-8),
    TensorBoard(histogram_freq=3, write_images=False,
                log_dir=logdir + '-stage2')]

m2.fit([Xt_t, Xt2_t], yt_t, epochs=num_epochs, validation_data=([Xs_t, Xs2_t], ys_t),
       callbacks=k_calls, verbose=2)

# %%

# Fine-tune by unfreezing the earlier model and using a tiiiny learning rate
k_calls[2] = TensorBoard(histogram_freq=3, write_images=False,
                         log_dir=logdir + '-finetune')
m2.trainable = True
m2.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-8),
           loss=['binary_crossentropy'],
           metrics=metrics)
m2.fit([Xt_t, Xt2_t], yt_t, epochs=num_epochs, validation_data=([Xs_t, Xs2_t], ys_t),
       callbacks=k_calls, verbose=2)

# %%

m2_preds = m2.predict([Xs_t, Xs2_t])
res_df['AugT_Win%'] = m2_preds[:, 0]
res_df['AugO_Win%'] = m2_preds[:, 1]
res_df['AugPredWin'] = [tids[n] if m2_preds[n, 0] > .5 else oids[n] for n in range(Xs_t.shape[0])]
res_df['auglogloss'] = -(ys_t[:, 0] * np.log(m2_preds[:, 0]) + ys_t[:, 1] * np.log(m2_preds[:, 1]))
aug_acc = sum(res_df['AugPredWin'] == res_df['TrueWin']) / res_df.shape[0]

print('\n\nRuns finished.')
print('\t\tBase\t\tAug')
print('Acc:\t\t{:.2f}\t{:.2f}'.format(base_acc * 100, aug_acc * 100))
print('Loss:\t{:.2f}\t\t{:.2f}'.format(res_df.mean()['logloss'], res_df.mean()['auglogloss']))

print('\nDumbest predictions:\n')
augdumb = np.sort(res_df['auglogloss'])
for n in augdumb[-5:]:
    row = res_df.loc[res_df['auglogloss'] == n]
    winperc = row['AugT_Win%'].values[0]
    winperc = winperc if winperc > .5 else 1 - winperc
    print(names[row['AugPredWin'].values[0]] + ' ({:.2f}) vs '.format(winperc) + names[
        row['TrueWin'].values[0]] + ' in {}'.format(row.index.get_level_values(1).values[0]))

# %%
# Double checking on tournament scores in our validation years
val_str = ''
val_str += '\nScores for training years:\n'
val_str += '\t\tESPN\t\tLogLoss\t\tAcc.\n'
for season in np.arange(2012, 2018):
    br = Bracket(season, True)
    br.run(m2, st_df, adv_tdf, [scale, scale_st2])
    val_str += '{}\t\t{}\t\t{:.2f}\t\t\t{:.2f}\n'.format(season, br.espn_score, br.loss, br.accuracy)

val_str += '\nScores for validation years:\n'
val_str += '\t\tESPN\t\tLogLoss\t\tAcc.\n'
for season in [2018, 2019]:
    br = Bracket(season, True)
    br.run(m2, st_df, adv_tdf, [scale, scale_st2])
    val_str += '{}\t\t{}\t\t{:.2f}\t\t\t{:.2f}\n'.format(season, br.espn_score, br.loss, br.accuracy)

print(val_str)

# %%
# Now we make predictions for the tournament of the year
subs = pd.read_csv('./data/MSampleSubmissionStage2.csv')
sub_df = pd.DataFrame(columns=['GameID'], data=np.arange(subs.shape[0]),
                      dtype=int)
for idx, row in subs.iterrows():
    tmp = row['ID'].split('_')
    sub_df.loc[idx, ['Season', 'TID', 'OID']] = [int(t) for t in tmp]
s2_df = sub_df.copy()
s2_df['TID'] = sub_df['OID']
s2_df['OID'] = sub_df['TID']
sub_df = sub_df.set_index(['GameID', 'Season', 'TID', 'OID'])
s2_df = s2_df.set_index(['GameID', 'Season', 'TID', 'OID'])
sub_df['Pivot'] = 1;
s2_df['Pivot'] = 1
pred_tdf = st.getTourneyStats(sub_df.append(s2_df), sdf)
pred_1 = rescale(st.getMatches(sub_df, st_df, diff=True).astype(np.float32), scale)
pred_2 = rescale(st.getMatches(sub_df, pred_tdf, diff=True).astype(np.float32), scale_st2)

sub_preds = m2.predict([pred_1, pred_2])
tids = sub_df.index.get_level_values(2)
oids = sub_df.index.get_level_values(3)
sub_df['T_Win%'] = sub_preds[:, 0]
sub_df['O_Win%'] = sub_preds[:, 1]
sub_df['PredWin'] = [names[tids[n]] if sub_preds[n, 0] > .5 else names[oids[n]] for n in range(sub_df.shape[0])]

b2021 = Bracket(2021)
b2021.run(m2, st_df, pred_tdf, [scale, scale_st2])
b2021.printTree('./submissions/tree' + runID + '.txt', val_str=val_str)

subs['Pred'] = sub_df['T_Win%'].values
subs.to_csv('./submissions/sub' + runID + '.csv', index=False)

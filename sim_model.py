#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:57:22 2021

sim_model

Simulates the games based on the idea that a team can be modeled
as a multivariate gaussian.

Alternative to nn_model.py.

@author: jeff
"""

import numpy as np
import pylab as plab
import pandas as pd
import statslib as st
import seaborn as sns
import eventlib as ev
from sklearn.decomposition import KernelPCA, TruncatedSVD
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tourney import Bracket
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers, regularizers
from keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
from datetime import datetime

sdf, sdf_t, sdf_d = st.arrangeFrame(scaling=None, noinfluence=True)
savdf = st.getSeasonalStats(sdf, strat='relelo')
tdf, tdf_t, tdf_d = st.arrangeTourneyGames()
adv_tdf = st.getTourneyStats(tdf, sdf) 
pdf = ev.getTeamRosters()
tsdf = pd.read_csv('./data/PlayerAnalysisData.csv').set_index(['Season', 'TID'])
scale_df = sdf.groupby(['Season', 'TID']).apply(lambda x: (x - x.mean()) / x.std())

#%%
inf_df = scale_df.groupby(['Season', 'OID']).mean()
inf_df = inf_df.drop(columns=[col for col in inf_df.columns if 'O_' in col])
inf_df = inf_df.reset_index().rename(columns={'OID': 'TID'}).set_index(['Season', 'TID'])
inf_df = inf_df.drop(columns=['DayNum', 'T_Rank', 'T_Elo'])
inf_df.columns = [col + 'Inf' for col in inf_df.columns]

#%%
total_df = st.merge(savdf, inf_df)

total_mu = total_df.groupby(['Season']).mean()
total_std = total_df.groupby(['Season']).std()
total_df = st.normalizeToSeason(total_df)

lin_df = sdf[['T_FGA', 'T_FGM', 'T_FGM3', 'T_FGA3', 'T_FTM', 'T_FTA', 
              'T_OR', 'T_DR', 'T_Ast', 'T_Stl', 'T_Blk']]
lin_mu = lin_df.groupby(['Season']).mean()
lin_std = lin_df.groupby(['Season']).std()
for idx, grp in lin_df.groupby(['Season']):
    lin_df.loc[grp.index] = (grp - grp.mean()) / grp.std()

#%%
inp = keras.Input(shape=(total_df.shape[1] * 2,))
out = layers.Dense(1600, kernel_regularizer=regularizers.l1(.1))(inp)
out = layers.Dense(1600, kernel_regularizer=regularizers.l1(.1))(out)
out = layers.Dense(1600, kernel_regularizer=regularizers.l1(.1))(out)
out = layers.Dense(1500, kernel_regularizer=regularizers.l1(.1))(out)
out = layers.Dense(1500, kernel_regularizer=regularizers.l1(.1))(out)
reg = layers.Dense(lin_df.shape[1],
                        name='output')(out)
model = keras.Model(inputs=inp, outputs=reg)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2),
              loss=['mean_squared_error'])

#%%
K.clear_session()

#MODEL STAGE TWO
logdir = "./logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
k_calls = [tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-4,
                    patience=15,
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=True),
            TensorBoard(histogram_freq=3, write_images=False,
                        log_dir=logdir)]
tdf_seas = list(set(total_df.index.get_level_values(0)))
good_seasons = [True if idx[1] in tdf_seas else False for idx, row in lin_df.iterrows()]
targets = lin_df.loc[good_seasons].sort_index()
adf = st.getMatches(lin_df.loc[good_seasons], total_df).sort_index()
Xt, Xs, yt, ys = train_test_split(adf, targets)
model.fit(Xt, yt, epochs=10, validation_data=(Xs, ys),
          callbacks=k_calls)

adf = st.getMatches(tdf, total_df)
res_df = pd.DataFrame(index=tdf.index, columns=lin_df.columns, data=model.predict(adf) * \
    lin_std.loc[season].values[None, :] + lin_mu.loc[season].values[None, :])
            
res_df = res_df.dropna().sort_index().astype(float).round().astype(int)
ntdf = tdf.loc[res_df.index, res_df.columns].sort_index()

res_df['Winner'] = tdf_t

#%%



    
    



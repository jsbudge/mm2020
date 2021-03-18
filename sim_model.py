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
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import oas
import matplotlib.pyplot as plt
from tourney import Bracket
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers, regularizers
from keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
from datetime import datetime

def shuffle(df):
    idx = np.random.permutation(np.arange(df.shape[0]))
    return df.iloc[idx], idx

def rescale(df, scaler):
    return pd.DataFrame(index=df.index, columns=df.columns,
                        data=scaler.transform(df))

sdf, sdf_t, sdf_d = st.arrangeFrame(scaling=None, noinfluence=True)
df1 = sdf.drop(columns=[col for col in sdf.columns if col[:2] != 'T_'])
df2 = sdf.drop(columns=[col for col in sdf.columns if col[:2] != 'O_'])
df2.columns = ['T_' + col[2:] for col in df2.columns]
tdf, tdf_t, tdf_d = st.arrangeTourneyGames()
scale = StandardScaler()

#%%

scale.fit(df1)
df1 = rescale(df1, scale)
df2 = rescale(df2, scale)

tsvd = TruncatedSVD(n_components=20)
tsvd.fit(df2)

df1 = pd.DataFrame(index=df1.index, data=tsvd.transform(df1))
df2 = pd.DataFrame(index=df2.index, data=tsvd.transform(df2))

ml_df = df1 - df2

#%%

Xt, Xs, yt, ys = train_test_split(ml_df, sdf_t)

scale.fit(Xt)
Xt = rescale(Xt, scale)
Xs = rescale(Xs, scale)

#%%

rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(Xt, yt)

#%%

k_mu = df1.groupby(['Season', 'TID']).mean()
k_std = pd.DataFrame(index=k_mu.index, columns=['COV'])
for idx, grp in df1.groupby(['Season', 'TID']):
    k_std.loc[idx, 'COV'] = oas(grp)[0]

#%%
print('Running simulations...')

sim_res = pd.DataFrame(index=tdf.index, columns=['T%', 'O%'])
for idx, row in tdf.iterrows():
    t_mu = k_mu.loc[(idx[1], idx[2])]
    t_cov = k_std.loc[(idx[1], idx[2])][0]
    t_gen = np.random.multivariate_normal(t_mu, t_cov, 100)
    o_mu = k_mu.loc[(idx[1], idx[3])]
    o_cov = k_std.loc[(idx[1], idx[3])][0]
    o_gen = np.random.multivariate_normal(o_mu, o_cov, 100)
    res = rfc.predict(t_gen - o_gen)
    sim_res.loc[idx, ['T%', 'O%']] = [sum(res) / 100, 1 - sum(res) / 100]
    
    




    
    



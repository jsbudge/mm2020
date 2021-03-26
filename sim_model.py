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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.covariance import oas
import matplotlib.pyplot as plt
from tourney import Bracket
from datetime import datetime
import multiprocessing as mp

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

tsvd = TruncatedSVD(n_components=30)
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
print('Fitting estimator...')
rfc = AdaBoostClassifier(n_estimators=250, learning_rate=1e-3)
rfc.fit(Xt, yt)

#%%

k_mu = df1.groupby(['Season', 'TID']).apply(lambda x: np.average(x, axis=0, weights=np.arange(x.shape[0])**2))
k_std = pd.DataFrame(index=k_mu.index, columns=['COV'])
for idx, grp in df1.groupby(['Season', 'TID']):
    k_std.loc[idx, 'COV'] = oas(grp)[0]

#%%
print('Running simulations...')

def sim_games(x):
    idx, row = x
    t_mu = k_mu.loc[(idx[1], idx[2])]
    t_cov = k_std.loc[(idx[1], idx[2])][0]
    t_gen = np.random.multivariate_normal(t_mu, t_cov, 100)
    o_mu = k_mu.loc[(idx[1], idx[3])]
    o_cov = k_std.loc[(idx[1], idx[3])][0]
    o_gen = np.random.multivariate_normal(o_mu, o_cov, 100)
    res = rfc.predict(t_gen - o_gen)
    return [sum(res) / 100, names[idx[2]], 
            1 - sum(res) / 100, names[idx[3]]]
    
    
names = st.loadTeamNames()
proc = mp.Pool(5)
data = [(idx, row) for idx, row in tdf.iterrows()]
sim_res = pd.DataFrame(index=tdf.index,
                       columns=['T%', 'TName', 'O%', 'OName'],
                       data=proc.map(sim_games, [(idx, row) for idx, row in tdf.iterrows()]))
#%%
n_sims = 400
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
sub_df['Pivot'] = 1; s2_df['Pivot'] = 1
    
sim_res = pd.DataFrame(index=tdf.index, columns=['T%', 'TName', 'O%', 'OName'])
for idx, row in sub_df.append(s2_df).iterrows():
    t_mu = k_mu.loc[(idx[1], idx[2])]
    t_cov = k_std.loc[(idx[1], idx[2])][0]
    t_gen = np.random.multivariate_normal(t_mu, t_cov, n_sims)
    o_mu = k_mu.loc[(idx[1], idx[3])]
    o_cov = k_std.loc[(idx[1], idx[3])][0]
    o_gen = np.random.multivariate_normal(o_mu, o_cov, n_sims)
    res = rfc.predict(t_gen - o_gen)
    sim_res.loc[idx, ['T%', 'TName', 'O%', 'OName']] = [1 - sum(res) / n_sims, names[idx[2]], 
                                                        sum(res) / n_sims, names[idx[3]]]

sim_res = sim_res.dropna()

#%%
sim_check = sim_res.drop(columns=['TName', 'OName'])
br2021 = Bracket(2021)
br2021.runWithFrame(sim_check)

    
    



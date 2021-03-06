# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 18:40:20 2021

@author: Jeff

Random Forest (and various other ensemble classifier) hyperparameter search.
Also doubling as a test for parallel processing.
"""

import numpy as np
import pandas as pd
import statslib as st
from tqdm import tqdm
from datetime import datetime
from tourney import Bracket, kerasWrapper
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, QuantileTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import log_loss
from sklearn.svm import NuSVC
import multiprocessing as mp

def shuffle(df):
    idx = np.random.permutation(np.arange(df.shape[0]))
    return df.iloc[idx], idx

def rescale(df, scaler):
    return pd.DataFrame(index=df.index, columns=df.columns,
                        data=scaler.transform(df))

def seasonalCV(X, y):
    seas = X.index.get_level_values(1)
    for yr in list(set(seas)):
        train = seas != yr
        test = seas == yr
        Xt, tidx = shuffle(X.loc[train])
        yt = y.loc[train].iloc[tidx]
        Xs, sidx = shuffle(X.loc[test])
        ys = y.loc[test].iloc[sidx]
        yield Xt, Xs, yt, ys, yr
        
def splitSeason(X, y, yr):
    seas = X.index.get_level_values(1)
    train = seas != yr
    test = seas == yr
    Xt, tidx = shuffle(X.loc[train])
    yt = y.loc[train].iloc[tidx]
    Xs, sidx = shuffle(X.loc[test])
    ys = y.loc[test].iloc[sidx]
    return Xt, Xs, yt, ys

print('Loading raw data...')
sdf, sdf_t, sdf_d = st.arrangeFrame(scaling=None, noinfluence=True)
#sdf_t.index = sdf.index
tdf, tdf_t, tdf_d = st.arrangeTourneyGames()
adv_tdf = st.getTourneyStats(tdf, sdf)
#st_df = st.getSeasonalStats(sdf, strat='gausselo')
st_df = pd.read_csv('./data/CongStats.csv').set_index(['Season', 'TID'])
scale = QuantileTransformer()
names = st.loadTeamNames()
runID = datetime.now().strftime("%Y%m%d-%H%M%S")
#%%
print('Transforming data...')
merge_df = st.merge(st_df, adv_tdf)
scale.fit(merge_df)
merge_df = rescale(merge_df, scale)
mu_sc = merge_df.mean()
std_sc = merge_df.std()
merge_df = (merge_df - mu_sc) / std_sc
ml_df = st.getMatches(tdf, merge_df, diff=True).sort_index()

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
tids = sub_df.index.get_level_values(2)
oids = sub_df.index.get_level_values(3)
pred_tdf = st.getTourneyStats(sub_df.append(s2_df), sdf)
b2021 = Bracket(2021)

#%%
print('Loading models...')
pred_feats = st.merge(st_df, pred_tdf)
p_df = (rescale(pred_feats, scale) - mu_sc) / std_sc
pred_1 = st.getMatches(sub_df, p_df, diff=True).astype(np.float32)
          
brackets = dict(zip(np.arange(2012, 2020), 
                    [Bracket(n, True) for n in np.arange(2012, 2020)]))

#%%
print('Running fitting and predictions...')
targets = tdf_t[ml_df.index]
mdl_df = pd.DataFrame()

#Sets of kernels with individual hyperparam optimization
    
def rfc_iter(L, x):
    Xt, Xs, yt, ys = splitSeason(ml_df, targets, x[2])
    model = kerasWrapper(RandomForestClassifier(n_estimators=x[0],
                                                criterion=x[1]))
    model.fit(Xt, yt)
    brackets[x[2]].run(model, merge_df, scaling=None)
    L.append([x[2], x[0], x[1], 'rfc', 
         brackets[x[2]].espn_score, 
         brackets[x[2]].loss, brackets[x[2]].accuracy])
    
def ada_iter(L, x):
    Xt, Xs, yt, ys = splitSeason(ml_df, targets, x[2])
    model = kerasWrapper(AdaBoostClassifier(n_estimators=x[0],
                                                learning_rate=x[1]))
    model.fit(Xt, yt)
    brackets[x[2]].run(model, merge_df, scaling=None)
    L.append([x[2], x[0], x[1], 'ada', 
         brackets[x[2]].espn_score, 
         brackets[x[2]].loss, brackets[x[2]].accuracy])
    
def bag_iter(L, x):
    Xt, Xs, yt, ys = splitSeason(ml_df, targets, x[2])
    model = kerasWrapper(BaggingClassifier(base_estimator=x[1][1], n_estimators=x[0]))
    model.fit(Xt, yt)
    brackets[x[2]].run(model, merge_df, scaling=None)
    L.append([x[2], x[0], x[1][0], 'bag', 
         brackets[x[2]].espn_score, 
         brackets[x[2]].loss, brackets[x[2]].accuracy])

#%%
print('RFC')
mdl_res = []
func_iter = [(n, crit, season) for n in np.arange(50, 1500, 400) \
             for crit in ['gini', 'entropy'] for season in np.arange(2012, 2020)]
    
with mp.Manager() as man:
    mdl_res = man.list()
    processes = []
    for f in func_iter:
        p = mp.Process(target=rfc_iter, args=(mdl_res, f,))
        processes.append(p)
        p.start()
    for proc in processes:
        proc.join()
    mdl_res = list(mdl_res)
    
mdl_df = mdl_df.append(pd.DataFrame(columns=['season', 'param1', 'param2', 'type', 'espn', 'loss', 'acc'],
                      data=mdl_res).set_index(['season', 'param1', 'param2', 'type']))

#%%
print('Ada')
mdl_res = []
func_iter = [(n, crit, season) for n in np.arange(50, 1500, 400) \
             for crit in [1e-5, 1e-4, 1e-2] for season in np.arange(2012, 2020)]
    
with mp.Manager() as man:
    mdl_res = man.list()
    processes = []
    for f in func_iter:
        p = mp.Process(target=ada_iter, args=(mdl_res, f,))
        processes.append(p)
        p.start()
    for proc in processes:
        proc.join()
    mdl_res = list(mdl_res)
    
mdl_df = mdl_df.append(pd.DataFrame(columns=['season', 'param1', 'param2', 'type', 'espn', 'loss', 'acc'],
                      data=mdl_res).set_index(['season', 'param1', 'param2', 'type']))

#%%
print('Bag')
mdl_res = []
func_iter = [(n, crit, season) for n in np.arange(10, 50, 20) \
             for crit in [('svc', NuSVC()), ('dtree', None)] for season in np.arange(2012, 2020)]
    
with mp.Manager() as man:
    mdl_res = man.list()
    processes = []
    for f in func_iter:
        p = mp.Process(target=bag_iter, args=(mdl_res, f,))
        processes.append(p)
        p.start()
    for proc in processes:
        proc.join()
    mdl_res = list(mdl_res)
    
mdl_df = mdl_df.append(pd.DataFrame(columns=['season', 'param1', 'param2', 'type', 'espn', 'loss', 'acc'],
                      data=mdl_res).set_index(['season', 'param1', 'param2', 'type']))

#%%
av_df = mdl_df.groupby(['param1', 'param2', 'type']).mean()

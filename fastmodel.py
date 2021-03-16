# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 18:40:20 2021

@author: Jeff

Quick Regressor because I didn't realize the deadline was
coming up so fast.
"""

import numpy as np
import pandas as pd
import statslib as st
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import log_loss

def rescale(df, scaler):
    return pd.DataFrame(index=df.index, columns=df.columns,
                        data=scaler.transform(df))

sdf, sdf_t, sdf_d = st.arrangeFrame(scaling=None, noinfluence=True)
sdf_t.index = sdf.index
tdf, tdf_t, tdf_d = st.arrangeTourneyGames()
adv_tdf = st.getTourneyStats(tdf, sdf)
av_df = st.getSeasonalStats(sdf, strat='gausselo')
scale = StandardScaler()
#%%

avt_df = st.getMatches(tdf, av_df, diff=True).astype(np.float32).dropna()
advt_df = st.getMatches(tdf, adv_tdf, diff=True).astype(np.float32).dropna()
ml_df = st.merge(avt_df, advt_df).drop(columns=['T_RoundRank'])
target = OneHotEncoder(sparse=False).fit_transform(tdf_t[ml_df.index].values.reshape((-1, 1)))

#%%

Xt, Xs, yt, ys = train_test_split(ml_df, target)
scale.fit(Xt)
Xt = rescale(Xt, scale)
Xs = rescale(Xs, scale)

for n_class in [50, 200, 750, 1400, 2000]:
    rfc = RandomForestClassifier(n_estimators=n_class)
    
    rfeats = RFECV(rfc, cv=KFold(shuffle=True)).fit(Xt, yt)
    
    rfc.fit(Xt[Xt.columns[rfeats.support_]], yt)
    spreds = np.array(rfc.predict_proba(Xs[Xt.columns[rfeats.support_]]))[0, :, :]
    
    #prep submissions
    subs = pd.read_csv('./data/MSampleSubmissionStage1.csv')
    preds = pd.DataFrame(index=subs.index, columns=['GameID', 'Season', 'TID', 'OID'])
    for idx, row in subs.iterrows():
        tids = row['ID'].split('_')
        preds.loc[idx, ['GameID', 'Season', 'TID', 'OID']] = \
            [idx, int(tids[0]), int(tids[1]), int(tids[2])]
            
    preds = preds.set_index(['GameID', 'Season', 'TID', 'OID'])
    pred_data = rescale(st.merge(st.getMatches(preds, av_df, diff=True),
                         st.getMatches(preds, adv_tdf, diff=True)).drop(columns=['T_RoundRank']),
                        scale)
    pred_data = pred_data[Xt.columns[rfeats.support_]]
    
    subs['Pred'] = rfc.predict_proba(pred_data)[0]
    subs.to_csv('sub_{}.csv'.format(n_class), index=False)




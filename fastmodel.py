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
from tourney import Bracket, kerasWrapper
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import log_loss

def rescale(df, scaler):
    return pd.DataFrame(index=df.index, columns=df.columns,
                        data=scaler.transform(df))

sdf, sdf_t, sdf_d = st.arrangeFrame(scaling=None, noinfluence=True)
#sdf_t.index = sdf.index
tdf, tdf_t, tdf_d = st.arrangeTourneyGames()
adv_tdf = st.getTourneyStats(tdf, sdf)
#av_df = st.getSeasonalStats(sdf, strat='gausselo')
st_df = pd.read_csv('./data/CongStats.csv').set_index(['Season', 'TID'])
scale = StandardScaler()
names = st.loadTeamNames()
#%%

avt_df = st.getMatches(tdf, st_df, diff=True).astype(np.float32).dropna()
advt_df = st.getMatches(tdf, adv_tdf, diff=True).astype(np.float32).dropna()
ml_df = st.merge(avt_df, advt_df)
target = OneHotEncoder(sparse=False).fit_transform(tdf_t[ml_df.index].values.reshape((-1, 1)))

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
pred_tdf = st.getTourneyStats(sub_df.append(s2_df), sdf)

#%%

Xt, Xs, yt, ys = train_test_split(ml_df, target)
scale.fit(Xt)
Xt = rescale(Xt, scale)
Xs = rescale(Xs, scale)

for n_class in [50]:
    rfc = RandomForestClassifier(n_estimators=n_class)
    rfc.fit(Xt, yt)
    
#%%
#Now we make predictions for the tournament of the year
pred_1 = st.getMatches(sub_df, st.merge(st_df, pred_tdf), diff=True).astype(np.float32)
model = kerasWrapper(rfc)

sub_preds = model.predict(rescale(pred_1, scale))
tids = sub_df.index.get_level_values(2)
oids = sub_df.index.get_level_values(3)
sub_df['T_Win%'] = sub_preds[:, 0]
sub_df['O_Win%'] = sub_preds[:, 1]
sub_df['PredWin'] = [names[tids[n]] if sub_preds[n, 0] > .5 else names[oids[n]] for n in range(sub_df.shape[0])]

b2021 = Bracket(2021)
b2021.run(model, pred_1, scaling=scale)
#b2021.printTree('./submissions/tree' + runID + '.txt')

subs['Pred'] = sub_df['T_Win%'].values
#subs.to_csv('./submissions/sub' + runID + '.csv', index=False)

#%%
#Double checking on tournament scores in our validation years
print('\nScores for validation years:')
print('\t\tESPN\t\tLogLoss\t\tAcc.')
for season in [2018, 2019]:
    br = Bracket(season, True)
    br.run(model, st.merge(st_df, adv_tdf), scale)
    print('{}\t\t{}\t\t{:.2f}\t\t\t{:.2f}'.format(season, br.espn_score, br.loss, br.accuracy))




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
from datetime import datetime
from tourney import Bracket, kerasWrapper
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.feature_selection import RFECV
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import KernelPCA
from sklearn.metrics import log_loss

def shuffle(df):
    idx = np.random.permutation(np.arange(df.shape[0]))
    return df.iloc[idx], idx

def rescale(df, scaler):
    return pd.DataFrame(index=df.index, columns=df.columns,
                        data=scaler.transform(df))

print('Loading raw data...')
sdf, sdf_t, sdf_d = st.arrangeFrame(scaling=None, noinfluence=True)
#sdf_t.index = sdf.index
tdf, tdf_t, tdf_d = st.arrangeTourneyGames()
adv_tdf = st.getTourneyStats(tdf, sdf)
st_df = st.getSeasonalStats(sdf, strat='gausselo')
#st_df = pd.read_csv('./data/CongStats.csv').set_index(['Season', 'TID'])
scale = StandardScaler()
names = st.loadTeamNames()
runID = datetime.now().strftime("%Y%m%d-%H%M%S")
#%%
print('Transforming data...')
kpca = KernelPCA(n_components=30)
merge_df = st.merge(st_df, adv_tdf)
k_df = pd.DataFrame(index=merge_df.index, data=kpca.fit_transform(merge_df))
ml_df = st.getMatches(tdf, k_df, diff=True).sort_index()

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
Xt, _ = shuffle(ml_df.loc(axis=0)[:, :2017, :, :])
yt = tdf_t[Xt.index].values
Xs = ml_df.loc(axis=0)[:, 2018:, :, :]
ys = tdf_t[Xs.index].values
scale.fit(Xt)
Xt = rescale(Xt, scale)
Xs = rescale(Xs, scale)
pred_feats = st.merge(st_df, pred_tdf)
p_df = pd.DataFrame(index=pred_feats.index, data=kpca.transform(pred_feats))
pred_1 = st.getMatches(sub_df, p_df, diff=True).astype(np.float32)
scale_pred_feats = rescale(pred_1, scale)

models = [RandomForestClassifier(n_estimators=500),
           RandomForestClassifier(n_estimators=1500),
           GaussianProcessClassifier(kernel=kernels.RBF(1.0))]
          
models = [kerasWrapper(m) for m in models]

#%%
print('Running fitting and predictions...')
#Now we make predictions for the tournament of the year
for idx, m in tqdm(enumerate(models)):
    m.fit(Xt, yt)
    val_str = ''
    sub_preds = m.predict(scale_pred_feats)
    sub_df['T_Win%'] = sub_preds[:, 0]
    sub_df['O_Win%'] = sub_preds[:, 1]
    sub_df['PredWin'] = [names[tids[n]] if sub_preds[n, 0] > .5 else names[oids[n]] for n in range(sub_df.shape[0])]
    val_str += '\nScores for validation years:\n'
    val_str += '\t\tESPN\t\tLogLoss\t\tAcc.\n'
    for season in [2018, 2019]:
        br = Bracket(season, True)
        br.run(m, k_df, scaling=scale)
        val_str += '{}\t\t{}\t\t{:.2f}\t\t\t{:.2f}\n'.format(season, br.espn_score, br.loss, br.accuracy)
    b2021.run(m, p_df, scaling=scale)
    b2021.printTree('./submissions/tree' + runID + '_{}.txt'.format(idx), val_str = val_str)
    subs['Pred'] = sub_df['T_Win%'].values
    subs.to_csv('./submissions/sub' + runID + '_{}.csv'.format(idx), index=False)

#%%
#Double checking on tournament scores in our validation years
print('\nScores for validation years:')
for idx, m in enumerate(models):
    print(f'Model {idx}')
    print('\t\tESPN\t\tLogLoss\t\tAcc.')
    for season in [2018, 2019]:
        br = Bracket(season, True)
        br.run(m, k_df, scaling=scale)
        print('{}\t\t{}\t\t{:.2f}\t\t\t{:.2f}'.format(season, br.espn_score, br.loss, br.accuracy))




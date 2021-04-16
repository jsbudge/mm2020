"""
The idea behind this model is to conglomerate
several different models into one:
1) a neural net classifier
2) A similar games selector
3) simulation models based on mean and variance
4) various other random forest and gaussian process classifiers
that seem to do well on the data
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
import time
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import multiprocessing as mp
from tqdm import tqdm
from pandarallel import pandarallel

import statslib as st
from tourney import Bracket


def rescale(df, scaler):
    return pd.DataFrame(index=df.index, columns=df.columns,
                        data=scaler.transform(df))


debug = True
scale = StandardScaler()
names = st.loadTeamNames()
k_calls = [tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=1e-3,
    patience=40,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=15,
        verbose=0,
        mode="auto",
        min_delta=5e-4,
        cooldown=5,
        min_lr=1e-9)]

print('Loading data...')
sdf, sdf_t, sdf_d = st.arrangeFrame(scaling=None, noinfluence=True)
sdf_t.index = sdf.index
tdf, tdf_t, tdf_d = st.arrangeTourneyGames()
adv_tdf = st.getTourneyStats(tdf, sdf)

# Possible games for the year 2021
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

# Tourney stats with roundrank so we can do some regression analysis
# Drop current year from sdf since we don't have tourney results for it yet
rrank = st.getTourneyStats(tdf, sdf.sort_index().loc(axis=0)[:, :2021, :, :], round_rank=True)

st_df = pd.read_csv('./data/CongStats.csv').set_index(['Season', 'TID'])

print('Regression for round rankings')
# Get stats for both differences between teams in games and team feature vectors
diff_df = st.getMatches(sdf, st_df, diff=True)
t1_df, t2_df = st.getMatches(sdf, st_df, diff=False)

# Merge things together to get averages for the season
reg_df = st_df.merge(rrank, right_on=['Season', 'TID'], left_on=['Season', 'TID'])
reg_target = OneHotEncoder(sparse=False).fit_transform(reg_df['T_RoundRank'].values.reshape((-1, 1)))

# Drop RoundRank because it has information leakage
reg_df = reg_df.drop(columns=['T_RoundRank'])

# Split into train and test sets, and scale sets to make zero mean and unit variance
Xt, Xs, yt, ys = train_test_split(reg_df, reg_target)
scale.fit(Xt)
Xt = rescale(Xt, scale)
Xs = rescale(Xs, scale)

# Layers of the model. We want this to be softmax
# output, so we have probabilities of getting to a
# round based on how similar teams did historically.
# Also, weight things since there are fewer champions
# than first round exits, by nature of the tourney.
inp = keras.Input(shape=(reg_df.shape[1],))
out = layers.Dense(reg_df.shape[1], activation='relu', name='init_dense',
                   kernel_regularizer=regularizers.l2(1e-3),
                   bias_regularizer=regularizers.l2(1e-3))(inp)
out = layers.Dropout(.5)(out)
out = layers.Dense(reg_df.shape[1], activation='relu', name='dense_1',
                   kernel_regularizer=regularizers.l2(1e-3),
                   bias_regularizer=regularizers.l2(1e-3))(out)
out = layers.Dropout(.5)(out)
smax = layers.Dense(reg_target.shape[1], activation='softmax',
                    name='output')(out)
reg_mdl = keras.Model(inputs=inp, outputs=smax)
reg_mdl.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss=['categorical_crossentropy'],
                metrics=['acc'])

print('Fitting model...')
# The callbacks given earlier are to adjust the learning rate during training for
# best results, and to end the training when no real progress is being made.
# We set the number of training epochs really high as a result - it's never actually
# going to train for 5000 epochs.
mdl_hist = reg_mdl.fit(Xt, yt, validation_data=(Xs, ys), epochs=5000, verbose=0, callbacks=k_calls,
                       class_weight={0: .94, 1: .53, 2: .76, 3: .88, 4: .94, 5: .97, 6: .98, 7: .98})

# We'll want this information for later
prob_df = pd.DataFrame(index=reg_df.index, columns=['1st4', 'R1', 'R64', 'R32', 'S16', 'E8', 'FF', 'CH'],
                       data=reg_mdl.predict(rescale(reg_df, scale)))

mdf_2021 = rescale(st_df.merge(pred_tdf, right_on=['Season', 'TID'], left_on=['Season', 'TID']), scale)
prob2021 = pd.DataFrame(index=pred_tdf.index, columns=['1st4', 'R1', 'R64', 'R32', 'S16', 'E8', 'FF', 'CH'],
                        data=reg_mdl.predict(mdf_2021))

# Similar game score - given the differences, find close matches
# and select winner based on that
tdf_gamedf = st.getMatches(tdf, st_df, diff=True)
tdf_gamedf = tdf_gamedf.loc[tdf_gamedf.index.get_level_values(0).duplicated(keep='first')]
trunc_diff = diff_df.loc[diff_df.index.get_level_values(0).duplicated(keep='first')]


# Note that in order to get parallel speedup, this can't be run in
# an IPython terminal, you'll get errors. Must be run via
# terminal, which kills debugging. Hence the flag.
def getSimilarGames(_list, x):
    sm_dist = (trunc_diff - x[1]).apply(np.linalg.norm, axis=1)
    sd_mu = sm_dist[sm_dist != 0.0].mean()
    sd_std = sm_dist[sm_dist != 0.0].std()
    sg_idx = trunc_diff.loc[sm_dist <= sm_dist[sm_dist != 0.0].min() + sd_std].index
    smatch_games = sdf.loc[sg_idx]
    sweights = 1 / (sm_dist.loc[sg_idx] + .01)
    swinperc = ((smatch_games['T_Score'] > smatch_games['O_Score']) * sweights).sum() / sweights.sum()
    _list.append((x[0], (sg_idx, sm_dist.min(), sd_mu, sd_std, swinperc)))


tdf_match = pd.DataFrame(index=tdf_gamedf.index,
                         columns=['MatchIdx', 'SimScore', 'SimMean', 'SimSigma', 'Win%'])
print('Running similar matches...')
if debug:
    for idx, row in tqdm(tdf_gamedf.iterrows()):
        dist = (trunc_diff - row).apply(np.linalg.norm, axis=1)
        d_mu = dist[dist != 0.0].mean()
        d_std = dist[dist != 0.0].std()
        g_idx = trunc_diff.loc[dist <= dist[dist != 0.0].min() + d_std].index
        tdf_match.loc[idx, 'MatchIdx'] = g_idx
        match_games = sdf.loc[g_idx]
        weights = 1 / (dist.loc[g_idx] + .01)
        winperc = ((match_games['T_Score'] > match_games['O_Score']) * weights).sum() / weights.sum()
        tdf_match.loc[idx, ['SimScore', 'SimMean', 'SimSigma', 'Win%']] = [dist.min(), d_mu, d_std, winperc]
else:
    match_res = []
    func_iter = [t for t in tdf_gamedf.iterrows()]
    with mp.Manager() as man:
        match_res = man.list()
        processes = []
        for f in func_iter:
            p = mp.Process(target=getSimilarGames, args=(match_res, f,))
            processes.append(p)
            p.start()
        for proc in processes:
            proc.join()
        match_res = list(match_res)
    for m in match_res:
        tdf_match.loc[m[0]] = m[1]

print('Running Gaussian Process Classification/Probability...')
# Check out various kernels
kernels = ['']
gpc = GaussianProcessClassifier(kernel='')




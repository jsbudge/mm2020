'''
The idea behind this model is to conglomerate
several different models into one:
1) a neural net classifier
2) A similar games selector
3) simulation models based on mean and variance
4) various other random forest and gaussian process classifiers
that seem to do well on the data
'''

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

print('Loading raw data...')
scale = StandardScaler()
names = st.loadTeamNames()

sdf, sdf_t, sdf_d = st.arrangeFrame(scaling=None, noinfluence=True)
sdf_t.index = sdf.index
tdf, tdf_t, tdf_d = st.arrangeTourneyGames()
adv_tdf = st.getTourneyStats(tdf, sdf)
st_df = pd.read_csv('./data/CongStats.csv').set_index(['Season', 'TID'])

diff_df = st.getMatches(sdf, st_df, diff=True).astype(np.float32).dropna()
tspl_df = st.getMatches(sdf, st_df, diff=False).astype(np.float32).dropna()
sdf = sdf.loc[diff_df.index]
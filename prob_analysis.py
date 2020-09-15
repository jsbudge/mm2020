#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 15:07:58 2020

@author: jeff

An ever-expanding statistical analysis of team, game, and seasonal
features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statslib as st
import framelib as fl
import featurelib as feat
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PowerTransformer, PolynomialFeatures
import sklearn.cluster as clustering
plt.close('all')

files = st.getFiles()
final_df = fl.arrangeFrame(files, scaling=PowerTransformer())
unscale_df = fl.arrangeFrame(files)[0]#.loc(axis=0)[:, 2015, :, :]
avdf = st.getSeasonalStats(unscale_df)

sdf = final_df[0]#.loc(axis=0)[:, 2015, :, :]
sdf = sdf.drop(columns=['T_Elo', 'O_Elo', 'T_Rank', 'O_Rank']) #Remove overall measures of team
ssdf = sdf.drop(columns=['T_Score', 'O_Score'])
tsdf = sdf[[s for s in sdf.columns if s[:2] == 'T_']]

#%%
'''
Initial exploration

We have a bunch of stats. Let's sort through them and remove those that
are redundant in some way.
'''
#First, let's try getting some offense indicators; i.e., how good a team is at
#generating offense.

corrs = np.corrcoef(tsdf.T)
tscore_coeff = corrs[tsdf.columns == 'T_Score', :].flatten()
off_cols = tsdf.columns[tscore_coeff > .5]
offsim = feat.getFeatureSimilarityMatrix(tsdf[off_cols])
plt.figure('Off. Feature Similarity')
plt.subplot(1, 3, 1)
sns.heatmap(offsim[0])
plt.title('Corr')
plt.subplot(1, 3, 2)
sns.heatmap(offsim[1])
plt.title('MInfo')
plt.subplot(1, 3, 3)
sns.heatmap(offsim[2])
plt.title('Comb')

#...And defense...
corrs = np.corrcoef(sdf.T)
oscore_coeff = corrs[sdf.columns == 'T_ScoreShift', :].flatten()
def_cols = sdf.columns[oscore_coeff < -.1]
defsim = feat.getFeatureSimilarityMatrix(sdf[def_cols])
plt.figure('Def. Feature Similarity')
plt.subplot(1, 3, 1)
sns.heatmap(defsim[0])
plt.title('Corr')
plt.subplot(1, 3, 2)
sns.heatmap(defsim[1])
plt.title('MInfo')
plt.subplot(1, 3, 3)
sns.heatmap(defsim[2])
plt.title('Comb')

#%%
'''
OFFENSIVE INDICATORS
'''

#Initial testing: we have our significant variables, let's try to reduce them to something useful.
eigdf = tsdf.drop(columns=['T_Score'])
U, sig, Vt = np.linalg.svd(eigdf, full_matrices=False)
plt.figure('Eigs')
plt.subplot(2, 1, 1)
plt.plot(sig)
plt.subplot(2, 1, 2)
plt.plot(np.gradient(sig))

ocorrs = np.corrcoef(U.T, unscale_df['T_Score'])[-1, :-1]
dcorrs = np.corrcoef(U.T, unscale_df['T_ScoreShift'])[-1, :-1]

plt.figure('Eig Corrs')
plt.plot(abs(ocorrs))
plt.plot(abs(dcorrs))

# plt.figure()
# plt.plot(U[0, :])
# plt.plot(eigdf.iloc[0].dot(np.linalg.pinv(Vt)).dot(np.linalg.pinv(np.diag(sig))))

oedf = pd.DataFrame(index=sdf.index, data=U[:, abs(ocorrs) > .1])
dedf = pd.DataFrame(index=sdf.index, data=U[:, abs(dcorrs) > .1])

for c in oedf.columns:
    hist, bins = np.histogram(oedf[c], bins='fd')
    plt.plot(bins[1:], hist)
    
#As gaussian random variables, we can get a mean and variance and find the outliers in every feature
oedf = (oedf - oedf.mean()) / oedf.std()
omeans = oedf.groupby(['Season', 'TID']).mean()
    

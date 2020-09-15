#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:51:31 2020

@author: jeff

FeatureLib

runs different algorithms and such on features. Needed a place for
all my ideas that didn't fit the other libraries' stated purpose.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from scipy.stats import hmean
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif, f_classif, \
    SelectPercentile, VarianceThreshold, f_regression, mutual_info_regression
from sklearn.metrics import log_loss, accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

def getFeatureInfo(X, y):
    ret = pd.DataFrame(columns=X.columns, index=['F_ANOVA', 'MINFO', 'CORR'])
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    ret.loc['F_ANOVA'] = hmean(np.array([f_classif(X.iloc[train], y.iloc[train])[0] for train, test in cv.split(X, y)]))
    ret.loc['MINFO'] = hmean(np.array([mutual_info_classif(X.iloc[train], y.iloc[train])[0] for train, test in cv.split(X, y)]))
    ret.loc['CORR'] = X.corrwith(y)
    return ret

def getFeatureSimilarityMatrix(X):
    temp = X.corr()
    mtemp = pd.DataFrame(index=temp.index, columns=temp.columns, data=0)
    comb = mtemp.copy()
    for n, col in enumerate(temp.columns):
        mx = X[col].values
        m_score = mutual_info_regression(X[X.columns[n:]], mx)
        mtemp.loc[col, X.columns[n:]] = m_score / m_score[0]
    mtemp = mtemp + mtemp.T
    #Scale to one
    comb = np.sqrt(temp**2 + mtemp**2) / np.sqrt(2)
    return temp, mtemp, comb

def classify(X, y, cl, cv=None):
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True)
    d = {}
    for c in cl:
        r = pd.DataFrame(columns=['Loss', 'Acc', 'AP', 'ROC'], index=np.arange(cv.n_splits+1))
        
        for n, (train, test) in enumerate(cv.split(X, y)):
            tshuff = np.random.permutation(len(train))
            sshuff = np.random.permutation(len(test))
            c[1].fit(X.iloc[train].iloc[tshuff], y.iloc[train].iloc[tshuff])
            res = c[1].predict_proba(X.iloc[test].iloc[sshuff])
            r.iloc[n] = [log_loss(y.iloc[test].iloc[sshuff].values, res[:, 1]), 
                         accuracy_score(y.iloc[test].iloc[sshuff].values, res[:, 1] > .5),
                         average_precision_score(y.iloc[test].iloc[sshuff].values, res[:, 1]),
                         roc_auc_score(y.iloc[test].iloc[sshuff].values, res[:, 1])]
        r.iloc[-1] = r.iloc[:n].mean()
        d[c[0]] = r
        try:
            d[c[0] + '_feat_im'] = c[1].feature_importances_
        except:
            d[c[0] + '_feat_im'] = 0
    return d

def proprietaryFeats(X):
    Xt = pd.DataFrame()
    #Tournament offense measure
    cols = ['T_Score', 'T_FG%', 'T_PPS', 'T_eFG%', 'T_TS%', 'T_OffRat', 'T_GameScore']
    means = np.array([ 69.49013453, 0.43763814, 0.99341058,
         0.49670529, 0.53726438, 109.89510644, 55.40504212])
    stds = np.array([11.9804552, 0.07539385, 0.17392086, 0.08696043,
        0.08189926, 16.45710437, 5.0516743 ])
    lambdas = np.array([0.57941522, 0.0207777, 0.37880789, 0.06679493,
       0.93635365, 1.00223, 0.49847729])
    #Remove mean and standard deviation
    for n, c in enumerate(cols):
        Xt[c] = (X[c] - means[n]) / stds[n]
        #Yeo-Johnson transform
        if lambdas[n] != 0:
            Xt.loc[Xt[c] >= 0, c] = ((Xt.loc[Xt[c] >= 0, c] + 1)**lambdas[n] - 1) / lambdas[n]
        else:
            Xt.loc[Xt[c] >= 0, c] = np.log10(Xt.loc[Xt[c] >= 0, c] + 1)
        if lambdas[n] != 2:
            Xt.loc[Xt[c] < 0, c] = -((-Xt.loc[Xt[c] < 0, c] + 1)**(2 - lambdas[n]) - 1) / (2 - lambdas[n])
        else:
            Xt.loc[Xt[c] < 0, c] = -np.log10(-Xt.loc[Xt[c] < 0, c] + 1)
    Xt['TourneyOffenseRat'] = np.mean(Xt, axis=1) * 50 / 3 + 50
    return Xt['TourneyOffenseRat']
        
class seasonalCV(object):
    def __init__(self, seasons):
        self.n_splits = len(seasons)
        self.seasons = seasons
        
    def split(self, X, y):
        for s in self.seasons:
            sn = X.index.get_level_values(1) == s
            train = np.arange(X.shape[0])[np.logical_not(sn)]; test = np.arange(X.shape[0])[sn]
            yield train, test
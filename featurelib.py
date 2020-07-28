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
    SelectPercentile, VarianceThreshold
from sklearn.metrics import log_loss, accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

def getFeatureInfo(X, y):
    ret = pd.DataFrame(columns=X.columns, index=['F_ANOVA', 'MINFO', 'CORR'])
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    ret.loc['F_ANOVA'] = hmean(np.array([f_classif(X.iloc[train], y.iloc[train])[0] for train, test in cv.split(X, y)]))
    ret.loc['MINFO'] = hmean(np.array([mutual_info_classif(X.iloc[train], y.iloc[train])[0] for train, test in cv.split(X, y)]))
    ret.loc['CORR'] = X.corrwith(y)
    return ret

def classify(X, y, cl, cv=None):
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True)
    d = {}
    for c in cl:
        r = pd.DataFrame(columns=['Loss', 'Acc', 'AP', 'ROC'], index=np.arange(cv.n_splits))
        
        for n, (train, test) in enumerate(cv.split(X, y)):
            tshuff = np.random.permutation(len(train))
            sshuff = np.random.permutation(len(test))
            c[1].fit(X.iloc[train].iloc[tshuff], y.iloc[train].iloc[tshuff])
            res = c[1].predict_proba(X.iloc[test].iloc[sshuff])
            r.iloc[n] = [log_loss(y.iloc[test].iloc[sshuff].values, res[:, 1]), 
                         accuracy_score(y.iloc[test].iloc[sshuff].values, res[:, 1] > .5),
                         average_precision_score(y.iloc[test].iloc[sshuff].values, res[:, 1]),
                         roc_auc_score(y.iloc[test].iloc[sshuff].values, res[:, 1])]
        d[c[0]] = r
        try:
            d[c[0] + '_feat_im'] = c[1].feature_importances_
        except:
            d[c[0] + '_feat_im'] = 0
    return d
        
class seasonalCV(object):
    def __init__(self, seasons):
        self.n_splits = len(seasons)
        self.seasons = seasons
        
    def split(self, X, y):
        for s in self.seasons:
            sn = X.index.get_level_values(1) == s
            train = np.arange(X.shape[0])[np.logical_not(sn)]; test = np.arange(X.shape[0])[sn]
            yield train, test
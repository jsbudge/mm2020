#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 12:34:56 2020

@author: jeff

Vector arrangement library, with functions for arranging everything how
I want it for exploration.
"""

import numpy as np
import pandas as pd
import statslib as st
from itertools import combinations, permutations

def arrangeFrame(files, scaling=None, noraw=False):
    ts = st.getGames(files['MRegularSeasonDetailedResults']).drop(columns=['NumOT', 'GLoc'])
    ty = ts['T_Score'] > ts['O_Score'] - 0
    if noraw:
        ts = st.joinFrame(ts[['GameID', 'Season', 'DayNum', 'TID', 'OID', 'T_Score', 'O_Score']],
                          st.getStats(ts))
    else:
        ts = st.joinFrame(ts, st.getStats(ts))
    ts = st.addRanks(ts)
    ts = st.addElos(ts)
    ts = ts.set_index(['GameID', 'Season', 'TID', 'OID'])
    #ts = st.joinFrame(ts, st.getInfluenceStats(ts)).set_index(['GameID', 'Season', 'TID', 'OID'])
    tsdays = ts['DayNum']
    ts = ts.drop(columns=['DayNum'])
    if scaling is not None:
        ts = st.normalizeToSeason(ts, scaler=scaling)
    return ts, ty, tsdays

def arrangeTourneyGames(files, noraw=False):
    tts = st.getGames(files['MNCAATourneyDetailedResults']).drop(columns=['NumOT', 'GLoc'])
    tty = tts['T_Score'] > tts['O_Score'] - 0
    if noraw:
        tts = st.joinFrame(tts[['GameID', 'Season', 'DayNum', 'TID', 'OID', 'T_Score', 'O_Score']],
                           st.getStats(tts)).set_index(['GameID', 'Season', 'TID', 'OID'])
    else:
        tts = st.joinFrame(tts, st.getStats(tts)).set_index(['GameID', 'Season', 'TID', 'OID'])
    ttsdays = tts['DayNum']
    tts = tts.drop(columns=['DayNum'])
    return tts, tty, ttsdays

def loadGames(sts, games):
    ttl = pd.DataFrame(index=games.index).merge(sts, 
                                            left_on=['Season', 'OID'], 
                                            right_on=['Season', 'TID'], 
                                            right_index=True)
    y = (games['T_Score'] - games['O_Score']) > 0
    X = ttl.groupby(['GameID']).diff().fillna(0) + ttl.groupby(['GameID']).diff(periods=-1).fillna(0)
    return X.sort_index(), y.sort_index()

def splitGames(X, y, split=None, as_frame=False):
    X = X.sort_index(); y = y.sort_index()
    if split is None:
        return X, y
    else:
        try:
            s = X.index.get_level_values(1) == split
            train = np.logical_not(s); test = s
        except:
            tr, te = next(split.split(X, y))
            train = [s in tr for s in range(X.shape[0])]
            test = [s in te for s in range(X.shape[0])]
        if as_frame:
            return X.loc[train], y[train], X.loc[test], y[test]
        else:
            return X.loc[train].values, y[train].values, X.loc[test].values, y[test].values
        
def getAllMatches(files, sts, season, transform=None):
    sd = pd.read_csv(files['MNCAATourneySeeds'])
    sd = sd.loc[sd['Season'] == season]['TeamID'].values
    teams = list(set(sd))
    matches = [[x, y] for (x, y) in permutations(teams, 2)]
    poss_games = pd.DataFrame(data=matches, columns=['TID', 'OID'])
    poss_games['Season'] = season; poss_games['GameID'] = np.arange(poss_games.shape[0])
    poss_games = poss_games.set_index(['GameID', 'Season', 'TID', 'OID'])
    ttl = pd.DataFrame(index=poss_games.index).merge(sts, 
                                            left_on=['Season', 'OID'], 
                                            right_on=['Season', 'TID'], 
                                            right_index=True).sort_index(level=0)
    ttw = pd.DataFrame(index=poss_games.index).merge(sts, 
                                            left_on=['Season', 'TID'], 
                                            right_on=['Season', 'OID'], 
                                            right_index=True).sort_index(level=0)
    if transform is None:
        return ttw - ttl
    else:
        return pd.DataFrame(index=ttw.index, 
                            columns=ttw.columns, 
                            data=transform.transform(ttw-ttl))

def merge(*args):
    df = None
    for d in args:
        if df is None:
            df = d
        else:
            df = df.merge(d, left_index=True, right_index=True)
    return df

def transformFrame(df, trans, fit=False):
    if fit:
        dt = trans.fit_transform(df)
        return pd.DataFrame(index=df.index, data=dt), trans
    else:
        return pd.DataFrame(index=df.index, data=trans.transform(df))

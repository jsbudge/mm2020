#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:34:16 2020

@author: jeff

Stores all of our classes and objects for data manipulation.
"""
import numpy as np
import pandas as pd
import statslib as st
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from itertools import combinations
from sklearn.pipeline import Pipeline, FeatureUnion

class Bracket(object):
    def __init__(self, season, files):
        seeds = pd.read_csv(files['MNCAATourneySeeds'])
        seeds = seeds.loc[seeds['Season'] == season]
        slots = pd.read_csv(files['MNCAATourneySlots'])
        slots = slots.loc[slots['Season'] == season]
        seedslots = pd.read_csv(files['MNCAATourneySeedRoundSlots']).rename(columns={'GameSlot': 'Slot'})
        structure = slots.merge(seedslots[['Slot', 'GameRound']], on='Slot')
        structure = structure.loc[np.logical_not(structure.duplicated(['Season', 'Slot'], keep='first'))].sort_values('GameRound')
        playin_seeds = []
        for idx, row in structure.iterrows():
            if row['GameRound'] == 0:
                structure.loc[idx, ['StrongSeed', 'WeakSeed']] = \
                    [seeds.loc[seeds['Seed'] == row['StrongSeed'], 'TeamID'].values[0],
                     seeds.loc[seeds['Seed'] == row['WeakSeed'], 'TeamID'].values[0]]
                playin_seeds.append(row['StrongSeed'][:3])
            elif row['GameRound'] == 1:
                if row['WeakSeed'] not in playin_seeds:
                    structure.loc[idx, ['StrongSeed', 'WeakSeed']] = \
                    [seeds.loc[seeds['Seed'] == row['StrongSeed'], 'TeamID'].values[0],
                     seeds.loc[seeds['Seed'] == row['WeakSeed'], 'TeamID'].values[0]]
                else:
                    structure.loc[idx, 'StrongSeed'] = \
                    seeds.loc[seeds['Seed'] == row['StrongSeed'], 'TeamID'].values[0]
        structure['Winner'] = 0; structure['StrongSeed%'] = 0; structure['WeakSeed%'] = 0
        truth = structure.copy()
        actual_games = pd.read_csv(files['MNCAATourneyCompactResults'])
        actual_games = actual_games.loc[actual_games['Season'] == season]
        for idx in range(structure.shape[0]):
            row = truth.iloc[idx]
            gm_res = actual_games.loc[np.logical_and(np.logical_or(actual_games['WTeamID'] == row['StrongSeed'],
                                                    actual_games['WTeamID'] == row['WeakSeed']),
                                                     np.logical_or(actual_games['LTeamID'] == row['StrongSeed'],
                                                    actual_games['LTeamID'] == row['WeakSeed']))]
            winner = gm_res['WTeamID'].values[0]
            truth.loc[truth['Slot'] == row['Slot'], 'Winner'] = winner
            if row['Slot'] in truth['StrongSeed'].values:
                truth.loc[truth['StrongSeed'] == row['Slot'], 'StrongSeed'] = winner
            else:
                truth.loc[truth['WeakSeed'] == row['Slot'], 'WeakSeed'] = winner
        self.season = season
        self.seeds = seeds
        self.structure = structure
        self.truth = truth
        self.tnames = st.loadTeamNames(files)
        self.isBuilt = False
        
    def __str__(self):
        order = ['W', 'X', 'Y', 'Z']
        match = [0, 0, 0, 0]; idx = 0
        mns = [[1, 8, 5, 4, 6, 3, 7, 2],
               [1, 4, 3, 2], [1, 2], [1]]
        if self.isBuilt:
            print_struct = self.structure.loc[self.structure['GameRound'] > 0].sort_values('Slot')
            ret = 'Model evaluated for {}. Score: {}, Loss: {:.2f}, Acc: {:.2f}%\n'.format(self.season,
                                                                                           self.espn_score,
                                                                                           self.loss,
                                                                                           self.accuracy * 100)
        else:
            print_struct = self.truth.loc[self.truth['GameRound'] > 0].sort_values('Slot')
            ret = '\nNo model evaluated.\n'
        ret += 'RND64\t\t\tRND32\t\t\tSWT16\t\t\tE8\t\t\tFF\t\t\tCH\n'
        for n in range(1, len(print_struct)+1):
            rnd = len(np.arange(6)[n % 2**np.arange(6) == 0])
            #nrnd = len(np.arange(6)[(n+1) % 2**np.arange(6) == 0])
            rndnme = 'R6CH'
            if rnd < 5:
                rndnme = 'R{}'.format(rnd) + order[idx] + str(mns[rnd-1][match[rnd-1]])
            elif rnd == 5:
                rndnme = 'R{}'.format(rnd) + order[idx] + order[idx+1]
            row = print_struct.loc[print_struct['Slot'] == rndnme]
            tmp = ''.join(['\t\t\t' for rng in range(rnd-1)])
            tmp += self.tnames[row['StrongSeed'].values[0]] + \
                ': {:.2f}, {:.2f} :'.format(row['StrongSeed%'].values[0], row['WeakSeed%'].values[0]) + \
                    self.tnames[row['WeakSeed'].values[0]] + '\n'
            tmp += ''.join(['\t\t\t' for rng in range(rnd-1)]) + \
                '(' + self.tnames[self.truth.loc[self.truth['Slot'] == rndnme, 'Winner'].values[0]] + ') ' + \
                    self.tnames[print_struct.loc[print_struct['Slot'] == rndnme, 'Winner'].values[0]] + '\n'
            ret += tmp
            if rnd < 5:
                    match[rnd-1] += 1
            else:
                idx = idx + 1
                match = [0, 0, 0, 0]
        return ret
        
    """
    run
    Runs through the tournament using data provided and the trained classifier
    provided.
    
    Params:
        classifier: sklearn model with predict() and predict_proba() function call.
        fc: FeatureCreator - prepped FeatureCreator.
    """
    def run(self, fc):
        for idx in range(self.structure.shape[0]):
            row = self.structure.iloc[idx]
            gm_res, prob = fc.classify(self.season, row['StrongSeed'], row['WeakSeed'])
            winner = row['StrongSeed'] if gm_res else row['WeakSeed']
            self.structure.loc[self.structure['Slot'] == row['Slot'], 
                               ['Winner', 'StrongSeed%', 'WeakSeed%']] = \
                                   [winner, prob[0][1], prob[0][0]]
            if row['Slot'] in self.structure['StrongSeed'].values:
                self.structure.loc[self.structure['StrongSeed'] == row['Slot'], 'StrongSeed'] = winner
            else:
                self.structure.loc[self.structure['WeakSeed'] == row['Slot'], 'WeakSeed'] = winner
        self.isBuilt = True
        success = (self.truth.loc[self.truth['GameRound'] > 0, 'Winner'] - \
                   self.structure.loc[self.structure['GameRound'] > 0, 'Winner']) == 0
        score = sum(2**(self.truth.loc[self.truth['GameRound'] > 0, 'GameRound'].values-1) * 10 * success)
        self.espn_score = score
        self.flat_score = sum(success)
        self.loss = log_loss(success, self.structure.loc[self.structure['GameRound'] > 0, 'StrongSeed%'].values)
        self.accuracy = sum(success) / self.structure.loc[self.structure['GameRound'] > 0].shape[0]
        
    """
    runAll
    Runs through every possible matchup for the season using data provided and the trained classifier
    provided.
    
    Params:
        classifier: sklearn model with predict() and predict_proba() function call.
        fc: FeatureCreator - prepped FeatureCreator.
        
    Returns:
        poss_games: DataFrame - frame with team IDs and probabilities of winning.
    """
    def runAll(self, classifier, fc):
        matches = [[x, y] for (x, y) in combinations(fc.getIndex(pd.Index([self.season])).index.get_level_values(1), 2)]
        poss_games = pd.DataFrame(data=matches, columns=['T', 'O'])
        poss_games['Season'] = self.season
        vector = fc.g_transform.transform(fc.getIndex(poss_games.set_index(['Season', 'T']).index).values - \
            fc.getIndex(poss_games.set_index(['Season', 'O']).index).values)
        gm_res = classifier.predict(vector)
        prob = classifier.predict_proba(vector)
        poss_games['Winner'] = [i[gm_res[n]] for n, i in enumerate(matches)]
        poss_games['W%'] = np.max(prob, axis=1)
        p1 = poss_games.copy()
        p1['T'], p1['O'] = poss_games['O'], poss_games['T']
        p1['W%'] = 1 - poss_games['W%']
        poss_games = poss_games.append(p1, ignore_index=True)
        return poss_games
        
    
    '''
    printTree
    Prints a pretty tournament tree to file with all the data you'd ever want.
    
    Params:
        fname: String - path to file
        
    Returns:
        True if successful
    '''
    def printTree(self, fname):
        try:
            with open(fname, 'w') as f:
                f.write(str(self))
            return True
        except:
            return False
        
'''
FeatureCreator
an object that stores created features and transforms for later use.

General usage
1) Call fc = FeatureCreator(files)
2) transform the feature vectors using feature_transform and an sklearn transform
3) Load games using loadGames and a getGames frame
4) transform the loaded games using game_transform and an sklearn transform
    
From here, you can load transformed games into a classifier or grab hypothetical games
using getGame or splitGames, which creates a training and testing set.
'''
class GameFrame(object):
    def __init__(self, files, frame):
        self.files = files
        self.tnames = st.loadTeamNames(files)
        self.frame = frame
        self.tourney_stats = st.getTourneyStats(frame.tdf, frame.sdf, files)
        self.loadGames(self.frame.tdf.index)
        self.clist = []
        self.tlist = []
        self.class_rank = []
        self.bc = None
        self.trans = None
        
    def getGameRank(self, season=None, tid=None):
        df = self.tourney_stats['GameRank']
        df = df.loc(axis=0)[season, :] if season is not None else df
        df = df.loc(axis=0)[:, tid] if tid is not None else df
        return df
        
    def get(self, season=None, tid=None, oid=None):
        df = self.X
        try:
            df = df.loc(axis=0)[:, season, :, :] if season is not None else df
            df = df.loc(axis=0)[:, :, tid, :] if tid is not None else df
            df = df.loc(axis=0)[:, :, :, oid] if oid is not None else df
            y = self.y[df.index]
            if df.shape[0] == 1:
                return df.values.reshape(1, -1), y.values
            else:
                return df.values, y.values
        except:
            #Game isn't in the X frame, we'll have to create it
            t1 = self.frame.sts.loc(axis=0)[season, tid].values
            t2 = self.frame.sts.loc(axis=0)[season, oid].values
            #Since the game hasn't been played for reals, y can be anything
            return self.trans.transform((t1 - t2).reshape(1, -1)), np.array([True])
        
    def add_t(self, ts):
        try:
            self.tlist = self.tlist + ts
        except:
            self.tlist.append(ts)
            
    def remove_t(self, name):
        for idx, n in enumerate(self.tlist):
            if n[0] == name:
                return self.tlist.pop(idx)
        return None
    
    def run_t(self):
        if np.any(self.tlist):
            self.trans = Pipeline(self.tlist)
            self.X = pd.DataFrame(index = self.X.index,
                                  data = self.trans.fit_transform(self.X, self.y))
    
    def add_c(self, ts):
        try:
            self.clist = self.clist + ts
        except:
            self.clist.append(ts)
        self.class_rank = [0 for n in self.clist]
            
    def remove_c(self, name):
        for idx, n in enumerate(self.clist):
            if n[0] == name:
                return self.clist.pop(idx)
        return None
    
    def run_c(self, Xt, yt, Xs, ys):
        if np.any(self.clist):
            print('===CLASSIFICATION RESULTS===\nClassifier\tLoss\tAccuracy')
            for n, c in enumerate(self.clist):
                c[1].fit(Xt, yt)
                loss = log_loss(ys, c[1].predict_proba(Xs))
                acc = sum(np.logical_and(ys, c[1].predict(Xs))) / len(ys) * 100
                self.class_rank[n] = acc
                print(c[0] + '\t{:.2f}\t{:.2f}'.format(loss, acc))
            output = [0] * len(self.clist)
            for i, x in enumerate(sorted(range(len(self.clist)), key=lambda y: self.clist[y])):
                if i == 0:
                    self.bc = self.clist[x]
                output[x] = i
            self.class_rank = output
        else:
            print('No classifiers loaded.')
            
    def classify(self, season=None, tid=None, oid=None):
        t = self.get(season, tid, oid)[0]
        return self.bc[1].predict(t), self.bc[1].predict_proba(t)
    
    def execute(self, Xt, yt, Xs, ys):
        self.run_t()
        self.run_c(Xt, yt, Xs, ys)
    
    def loadGames(self, game_idx, y=None):
        ttl = pd.DataFrame(index=game_idx).merge(self.frame.sts, 
                                                        left_on=['Season', 'OID'], 
                                                        right_on=['Season', 'TID'], 
                                                        right_index=True).sort_index(level=0)
        X = ttl.groupby(['GameID']).diff().fillna(0) + ttl.groupby(['GameID']).diff(periods=-1).fillna(0)
        if y is None:
            try:
                y = self.frame.tdf_y.loc[X.index]
            except:
                y = self.frame.sdf_y.loc[X.index]
        self.X = X; self.y = y
        return X, y
    
    def splitGames(self, split=None, as_frame=False):
        if split is None:
            return self.X, self.y
        else:
            try:
                s = self.X.index.get_level_values(1) == split
                self.train = np.logical_not(s); self.test = s
            except:
                train, test = next(split.split(self.X, self.y))
                self.train = [s in train for s in range(self.X.shape[0])]
                self.test = [s in test for s in range(self.X.shape[0])]
            if as_frame:
                return self.X.loc[self.train], self.y[self.train], self.X.loc[self.test], self.y[self.test]
            else:
                return self.X.loc[self.train].values, self.y[self.train].values, self.X.loc[self.test].values, self.y[self.test].values
        
class FeatureFrame(object):
    def __init__(self, files, strat='rank', scaling=None):
        ts = st.getGames(files['MRegularSeasonDetailedResults']).drop(columns=['NumOT'])
        ts = st.addRanks(ts)
        ts = st.addElos(ts)
        ts = st.joinFrame(ts, st.getStats(ts))
        ts = st.joinFrame(ts, st.getInfluenceStats(ts)).set_index(['GameID', 'Season', 'TID', 'OID'])
        tsdays = ts['DayNum']
        ts = ts.drop(columns=['DayNum', 'Unnamed: 0'])
        ty = ts['T_Score'] > ts['O_Score'] - 0
        tts = st.getGames(files['MNCAATourneyDetailedResults']).drop(columns=['NumOT']).set_index(['GameID', 'Season', 'TID', 'OID'])
        ttsdays = tts['DayNum']
        tts = tts.drop(columns=['DayNum'])
        tty = tts['T_Score'] > tts['O_Score'] - 0
        seasonal = st.getSeasonalStats(ts, seasonal_only=True)
        if scaling is not None:
            ts = st.normalizeToSeason(ts, scaler=scaling)
        self.files = files
        self.strat = strat
        self.tnames = st.loadTeamNames(files)
        self.tdf = tts
        self.sdf = ts
        self.sdf_y = ty
        self.tdf_y = tty
        self.sdf_d = tsdays
        self.tdf_d = ttsdays
        self.seasonal = seasonal
        self.actions = []
        
    def add_t(self, ts):
        try:
            self.actions = self.actions + ts
        except:
            self.actions.append(ts)
            
    def remove_t(self, name):
        for idx, n in enumerate(self.actions):
            if n[0] == name:
                return self.actions.pop(idx)
        return None
                
    def add_f(self, fname, feature):
        self.sdf[fname] = feature(self.sdf)
        
    def remove_f(self, fname):
        self.sdf = self.sdf.drop(columns=[fname])
        
    def get_f(self, fname, is_season=True):
        if is_season:
            return self.sdf[fname]
        else:
            return self.tdf[fname]
        
    def get(self, season=None, tid=None, oid=None, is_season=True):
        df = self.sdf if is_season else self.tdf
        df = df.loc(axis=0)[:, season, :, :] if season is not None else df
        df = df.loc(axis=0)[:, :, tid, :] if tid is not None else df
        df = df.loc(axis=0)[:, :, :, oid] if oid is not None else df
        return df
        
    def execute(self, season=None, y_fit=None, on_avs=False):
        y = y_fit if y_fit is not None else self.sdf_y
        if np.any(self.actions):
            pipe = Pipeline(self.actions)
            if not on_avs:
                self.sdf = pd.DataFrame(data=pipe.fit_transform(self.sdf, y),
                                        index = self.sdf.index,
                                        columns=self.sdf.columns)
        if season is not None:
            sts = st.getSeasonalStats(self.sdf.loc(axis=0)[:, season, :, :], strat=self.strat)
        else:
            sts = st.getSeasonalStats(self.sdf, strat=self.strat)
        sts[self.seasonal.columns] = self.seasonal.loc[sts.index]
        if on_avs:
            sts = pd.DataFrame(data=pipe.fit_transform(sts, y),
                                    index = sts.index,
                                    columns=sts.columns)
        self.sts = sts
        return sts

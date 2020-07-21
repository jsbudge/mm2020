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
    def run(self, classifier, fc):
        for idx in range(self.structure.shape[0]):
            row = self.structure.iloc[idx]
            vector = fc.getGame(self.season, row['StrongSeed'],
                                 row['WeakSeed'])
            gm_res = classifier.predict(vector)
            prob = classifier.predict_proba(vector)
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
class FeatureCreator(object):
    def __init__(self, files, strat='rank', transform=None, scaling=None, rc=False):
        self.files = files
        self.tnames = st.loadTeamNames(files)
        self.scaler = scaling
        self.average_strat = strat
        self.f_transform = transform
        self.g_transform = None
        self.init_sts = None
        self.reload(rc=rc)
        
    def reload(self, rc=False):
        if self.init_sts is None:
            ts = st.getGames(self.files['MRegularSeasonDetailedResults']).drop(columns=['NumOT'])
            ts = st.addRanks(ts)
            ts = st.addElos(ts)
            ts = st.joinFrame(ts, st.getStats(ts))
            ts = st.joinFrame(ts, st.getInfluenceStats(ts, recalc=rc))
            ttu = st.getGames(self.files['MNCAATourneyDetailedResults'])
            ttstats = st.getTourneyStats(ttu, ts, self.files)
            sts = st.getSeasonalStats(ts, strat=self.average_strat, recalc=rc)
            sts = sts.merge(ttstats[['T_FinalRank', 'T_FinalElo', 'T_Seed']], left_index=True, right_index=True)
            self.gamestats = ttstats[['GameRank', 'AdjGameRank']]
            self.init_sts = sts.copy()
        else:
            sts = self.init_sts.copy()
        if self.scaler is not None:
            sts = st.normalizeToSeason(sts, scaler=self.scaler)
        if self.f_transform is not None:
            sts = pd.DataFrame(index=sts.index, data=self.f_transform.transform(sts))
        self.avdf = sts
        
    def getGameRank(self):
        return self.gamestats['GameRank']
        
    def reAverage(self, strat):
        self.average_strat = strat
        self.init_sts = None
        self.reload()
        
    def feature_transform(self, transform, y=None):
        self.f_transform = transform.fit(self.avdf, y)
        self.reload()
        
    def game_transform(self, transform):
        self.g_transform = transform.fit(self.X, self.y)
        self.X = pd.DataFrame(index=self.X.index, data=self.g_transform.transform(self.X))
        
    def reScale(self, scaling):
        self.scaler = scaling
        self.reload()
        
    def get(self, season, tid):
        return self.avdf.loc[(season, tid), :]
    
    def getIndex(self, idx):
        return self.avdf.loc[idx]
    
    def getGame(self, season, tid, oid):
        try:
            return self.X.loc[(season, tid, oid)].values.reshape(1, -1)
        except:
            return self.g_transform.transform((self.avdf.loc[(season, tid)].values - self.avdf.loc[(season, oid)].values).reshape(1, -1))
    
    def loadGames(self, gameframe):
        ttw = self.getIndex(pd.MultiIndex.from_frame(gameframe[['Season', 'TID']]))
        ttl = self.getIndex(pd.MultiIndex.from_frame(gameframe[['Season', 'OID']]))
        X = pd.DataFrame(ttw.values - ttl.values)
        X['Season'] = gameframe['Season'].values
        X['TID'] = gameframe['TID'].values
        X['OID'] = gameframe['OID'].values
        X = X.set_index(['Season', 'TID', 'OID'])
        y = (gameframe['T_Score'] - gameframe['O_Score'] > 0).values - 0
        self.X = X; self.y = y
        return X, y
    
    def loadAndTransformGames(self, gameframe, f_transform, g_transform, y=None):
        self.feature_transform(f_transform, y)
        self.loadGames(gameframe)
        self.game_transform(g_transform)
    
    def splitGames(self, split=None):
        if split is None:
            return self.X, self.y
        else:
            if type(split) == int:
                s = self.X.index.get_level_values(0) == split
                self.train = np.logical_not(s); self.test = s
            else:
                train, test = next(split.split(self.X, self.y))
                self.train = [s in train for s in range(self.X.shape[0])]
                self.test = [s in test for s in range(self.X.shape[0])]
            return self.X.loc[self.train], self.y[self.train], self.X.loc[self.test], self.y[self.test]

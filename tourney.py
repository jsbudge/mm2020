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
        df: DataFrame - frame with index of TID, containing the feature vectors
                        for the teams that are playing.
        classifier: sklearn model with predict() and predict_proba() function call.
    """
    def run(self, classifier, fc):
        for idx in range(self.structure.shape[0]):
            row = self.structure.iloc[idx]
            vector = (fc.get(self.season, row['StrongSeed']) -\
                      fc.get(self.season, row['WeakSeed'])).values.reshape(1, -1)
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
        
class FeatureCreator(object):
    def __init__(self, files, strat='rank', transform=None, scaling=None):
        self.files = files
        self.tnames = st.loadTeamNames(files)
        self.scaler = scaling
        self.average_strat = strat
        self.transform = None
        self.init_sts = None
        self.reload()
        
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
            self.init_sts = sts.copy()
        else:
            sts = self.init_sts.copy()
        sts = st.normalizeToSeason(sts, scaler=self.scaler)
        if self.transform is not None:
            sts = pd.DataFrame(index=sts.index, data=self.transform.transform(sts))
        self.avdf = sts
        
    def reAverage(self, strat):
        self.average_strat = strat
        self.init_sts = None
        self.reload()
        
    def reTransform(self, transform):
        self.transform = transform
        self.reload()
        
    def reScale(self, scaling):
        self.scaler = scaling
        self.reload()
        
    def get(self, season, tid):
        return self.avdf.loc[(season, tid), :]
    
    def getIndex(self, idx):
        return self.avdf.loc[idx]
    
    def splitGames(self, gameframe, split=None):
        ttw = self.getIndex(pd.MultiIndex.from_frame(gameframe[['Season', 'TID']]))
        ttl = self.getIndex(pd.MultiIndex.from_frame(gameframe[['Season', 'OID']]))
        X = pd.DataFrame(ttw.values - ttl.values)
        y = (gameframe['T_Score'] - gameframe['O_Score'] > 0).values - 0
        if split is None:
            return X, y
        else:
            if type(split) == int:
                s = ttw.index.get_level_values(0) == split
                return X.loc[np.logical_not(s)], y[np.logical_not(s)], X.loc[s], y[s]
            train, test = next(split.split(X, y))
            return X.iloc[train], y[train], X.iloc[test], y[test]

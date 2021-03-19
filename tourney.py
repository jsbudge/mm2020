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
from statslib import getAllMatches
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from itertools import combinations
from sklearn.pipeline import Pipeline, FeatureUnion

def rescale(df, scaler):
    return pd.DataFrame(index=df.index, columns=df.columns,
                        data=scaler.transform(df))

class Bracket(object):
    def __init__(self, season, use_results=False):
        if use_results:
            results = pd.read_csv('./data/MNCAATourneyCompactResults.csv')
            results = results.loc[results['Season'] == season]
        else:
            results = None
        seeds = pd.read_csv('./data/MNCAATourneySeeds.csv')
        seeds = seeds.loc[seeds['Season'] == season]
        slots = pd.read_csv('./data/MNCAATourneySlots.csv')
        slots = slots.loc[slots['Season'] == season]
        seedslots = pd.read_csv('./data/MNCAATourneySeedRoundSlots.csv').rename(columns={'GameSlot': 'Slot'})
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
        if results is not None:
            truth = structure.copy()
            for idx in range(structure.shape[0]):
                row = truth.iloc[idx]
                gm_res = results.loc[np.logical_and(np.logical_or(results['WTeamID'] == row['StrongSeed'],
                                                        results['WTeamID'] == row['WeakSeed']),
                                                         np.logical_or(results['LTeamID'] == row['StrongSeed'],
                                                        results['LTeamID'] == row['WeakSeed']))]
                winner = gm_res['WTeamID'].values[0]
                truth.loc[truth['Slot'] == row['Slot'], 'Winner'] = winner
                if row['Slot'] in truth['StrongSeed'].values:
                    truth.loc[truth['StrongSeed'] == row['Slot'], 'StrongSeed'] = winner
                else:
                    truth.loc[truth['WeakSeed'] == row['Slot'], 'WeakSeed'] = winner
        else:
            truth = None
        pg = pd.DataFrame(columns=['TID', 'OID'],
                          data=combinations(seeds['TeamID'], 2))
        pg['Season'] = season
        pg['GameID'] = pg.index.values
        pg2 = pg.copy()
        pg2['TID'] = pg['OID']; pg2['OID'] = pg['TID']
        pg = pg.append(pg2)
        self.poss_games = pg.set_index(['GameID', 'Season', 'TID', 'OID'])
        self.truth = truth
        self.season = season
        self.seeds = seeds
        self.structure = structure
        self.tnames = st.loadTeamNames()
        self.isBuilt = False
        self.hasResults = True if results is not None else False
        self.espn_score = 0
        self.flat_score = 0
        self.loss = 0
        self.accuracy = 0
        
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
        elif self.hasResults:
            print_struct = self.truth.loc[self.truth['GameRound'] > 0].sort_values('Slot')
            ret = '\nNo model evaluated.\n'
        else:
            print_struct = self.structure.loc[self.structure['GameRound'] > 0].sort_values('Slot')
            ret = '\nNo model evaluated and no results loaded.\n'
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
            try:
                sseed = self.tnames[row['StrongSeed'].values[0]]
            except:
                sseed = row['StrongSeed'].values[0]
            try:
                wseed = self.tnames[row['WeakSeed'].values[0]]
            except:
                wseed = row['WeakSeed'].values[0]
            tmp += sseed + \
                ': {:.2f}, {:.2f} :'.format(row['StrongSeed%'].values[0], row['WeakSeed%'].values[0]) + \
                    wseed + '\n'
            if self.hasResults:
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
        classifier: keras model with two softmax outputs.
        feats: DataFrame - frame of team features to combine.
        add_feats: DataFrame - for if we want a classifier with multiple inputs
        scaling: sklearn Scaler - single instance of a scaler to use with
            feats. If using add_feats, a list of two scalers.
    """
    def run(self, classifier, feats, add_feats=None, scaling=None):
        bones = self.structure
        ag = st.getMatches(self.poss_games, feats, diff=True)
        ag2 = st.getMatches(self.poss_games, add_feats, diff=True) if add_feats is not None else None
        if scaling is not None:
            if ag2 is not None:
                ag = rescale(ag, scaling[0])
                ag2 = rescale(ag2, scaling[1])
            else:
                ag = rescale(ag, scaling)
        if ag2 is not None:
            gm = pd.DataFrame(index=ag.index, data=classifier.predict([ag, ag2]))
        else:
            gm = pd.DataFrame(index=ag.index, data=classifier.predict(ag))
        for idx in range(bones.shape[0]):
            row = bones.iloc[idx]
            gm_res = gm.loc(axis=0)[:, self.season, row['StrongSeed'], row['WeakSeed']].values[0]
            winner = row['StrongSeed'] if gm_res[0] > .5 else row['WeakSeed']
            bones.loc[bones['Slot'] == row['Slot'], 
                               ['Winner', 'StrongSeed%', 'WeakSeed%']] = \
                                   [winner, gm_res[0], gm_res[1]]
            if row['Slot'] in bones['StrongSeed'].values:
                bones.loc[bones['StrongSeed'] == row['Slot'], 'StrongSeed'] = winner
            else:
                bones.loc[bones['WeakSeed'] == row['Slot'], 'WeakSeed'] = winner
        self.isBuilt = True
        if self.hasResults:
            success = (self.truth.loc[self.truth['GameRound'] > 0, 'Winner'] - \
                       bones.loc[bones['GameRound'] > 0, 'Winner']) == 0
            score = sum(2**(self.truth.loc[self.truth['GameRound'] > 0, 'GameRound'].values-1) * 10 * success)
            self.espn_score = score
            self.flat_score = sum(success)
            self.loss = log_loss(success, bones.loc[bones['GameRound'] > 0, 'StrongSeed%'].values)
            self.accuracy = sum(success) / bones.loc[bones['GameRound'] > 0].shape[0]
        self.classifier = classifier
        
    """
    run
    Runs through the tournament using data provided and the trained classifier
    provided.
    
    Params:
        classifier: keras model with two softmax outputs.
        feats: DataFrame - frame of team features to combine.
        add_feats: DataFrame - for if we want a classifier with multiple inputs
        scaling: sklearn Scaler - single instance of a scaler to use with
            feats. If using add_feats, a list of two scalers.
    """
    def runWithFrame(self, gm):
        bones = self.structure
        for idx in range(bones.shape[0]):
            row = bones.iloc[idx]
            gm_res = gm.loc(axis=0)[:, self.season, row['StrongSeed'], row['WeakSeed']].values[0]
            winner = row['StrongSeed'] if gm_res[0] > .5 else row['WeakSeed']
            bones.loc[bones['Slot'] == row['Slot'], 
                               ['Winner', 'StrongSeed%', 'WeakSeed%']] = \
                                   [winner, gm_res[0], gm_res[1]]
            if row['Slot'] in bones['StrongSeed'].values:
                bones.loc[bones['StrongSeed'] == row['Slot'], 'StrongSeed'] = winner
            else:
                bones.loc[bones['WeakSeed'] == row['Slot'], 'WeakSeed'] = winner
        self.isBuilt = True
        if self.hasResults:
            success = (self.truth.loc[self.truth['GameRound'] > 0, 'Winner'] - \
                       bones.loc[bones['GameRound'] > 0, 'Winner']) == 0
            score = sum(2**(self.truth.loc[self.truth['GameRound'] > 0, 'GameRound'].values-1) * 10 * success)
            self.espn_score = score
            self.flat_score = sum(success)
            self.loss = log_loss(success, bones.loc[bones['GameRound'] > 0, 'StrongSeed%'].values)
            self.accuracy = sum(success) / bones.loc[bones['GameRound'] > 0].shape[0]
        
    """
    getProbabilities
    Runs through every possible matchup for the season using data provided and the trained classifier
    provided.
    
    Params:
        classifier: keras model with two softmax outputs.
        feats: DataFrame - frame of team features to combine.
        add_feats: DataFrame - for if we want a classifier with multiple inputs
        scaling: sklearn Scaler - single instance of a scaler to use with
            feats. If using add_feats, a list of two scalers.
        
    Returns:
        gm: DataFrame - frame with team IDs and probabilities of winning.
    """
    def getProbabilites(self, classifier, feats, add_feats=None, scaling=None):
        ag = st.getMatches(self.poss_games, feats, diff=True)
        ag2 = st.getMatches(self.poss_games, add_feats, diff=True) if add_feats is not None else None
        if scaling is not None:
            if ag2 is not None:
                ag = rescale(ag, scaling[0])
                ag2 = rescale(ag2, scaling[1])
            else:
                ag = rescale(ag, scaling)
        if ag2 is not None:
            gm = pd.DataFrame(index=ag.index, 
                              data=classifier.predict([ag, ag2]),
                              columns=['T%', 'O%'])
        else:
            gm = pd.DataFrame(index=ag.index, data=classifier.predict(ag),
                              columns=['T%', 'O%'])
        self.classifier = classifier
        return gm
        
    
    '''
    printTree
    Prints a pretty tournament tree to file with all the data you'd ever want.
    
    Params:
        fname: String - path to file
        
    Returns:
        True if successful
    '''
    def printTree(self, fname, val_str=None):
        try:
            with open(fname, 'w') as f:
                f.write(str(self))
                if self.isBuilt:
                    self.classifier.summary(print_fn=lambda x: f.write(x + '\n'))
                if val_str is not None:
                    f.write(val_str)
            return True
        except:
            return False
        
        
class kerasWrapper(object):
    def __init__(self, model, reg=False):
        self.isRegression = reg
        self.model = model
        
    def predict(self, data):
        if self.isRegression:
            return st.getPointSpreadProbability(self.model.predict(data))
        else:
            return self.model.predict_proba(data)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def summary(self, print_fn=None):
        d = self.model.__dict__
        ret = ''
        for k in d:
            if type(d[k]) != list:
                ret += k + ': ' + str(d[k]) + '\n'
        print_fn(ret)
        
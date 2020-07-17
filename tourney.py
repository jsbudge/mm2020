#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:34:16 2020

@author: jeff

Stores all of our classes and objects for data manipulation.
"""
import numpy as np
import pandas as pd
from statslib import loadTeamNames
from sklearn.metrics import log_loss

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
        self.tnames = loadTeamNames(files)
        self.isBuilt = False
        
    def __str__(self):
        order = ['W', 'X', 'Y', 'Z']
        match = [0, 0, 0, 0]; idx = 0
        mns = [[1, 8, 5, 4, 6, 3, 7, 2],
               [1, 4, 3, 2], [1, 2], [1]]
        if self.isBuilt:
            print_struct = self.structure.loc[self.structure['GameRound'] > 0].sort_values('Slot')
            ret = 'Model evaluated for {}. Score: {}, Loss: {:.2f}, Acc: {:.2f}%\n'.format(self.season,
                                                                                           self.score,
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
    def run(self, df, classifier, transformer=None):
        for idx in range(self.structure.shape[0]):
            row = self.structure.iloc[idx]
            if transformer is not None:
                vector = transformer.transform((df.loc[row['StrongSeed']] - \
                                        df.loc[row['WeakSeed']]).values.reshape(1, -1))
            else:
                vector = (df.loc[row['StrongSeed']] - df.loc[row['WeakSeed']]).values.reshape(1, -1)
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
        self.score = score
        self.loss = log_loss(success, self.structure.loc[self.structure['GameRound'] > 0, 'StrongSeed%'].values)
        self.accuracy = sum(success) / self.structure.loc[self.structure['GameRound'] > 0].shape[0]
    
    def printTree(self, fname):
        try:
            with open(fname, 'w') as f:
                f.write(str(self))
            return True
        except:
            return False
        
            
                
        

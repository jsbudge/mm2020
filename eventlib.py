# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 17:56:30 2020

@author: Jeff

Events library

Takes the Events and converts them into something a little more palatable.
Mostly, rosters and coach data.
"""

import numpy as np
import pandas as pd
import tqdm as tqdm

def getRosters(files, season):
    events = pd.read_csv(files['MEvents{}'.format(season)])
    players = pd.read_csv(files['MPlayers'])
    rdata = pd.DataFrame(columns=['TID', 'OID', 'DayNum', 'Season'])
    pdata = pd.DataFrame(columns=['PlayerID', 'GameID', 'Ast', 'Blk', 'Stl', 'Foul', 'Tech', 'TO', 'OR', 'DR', 'FTM', 'FTA',
                                  'FGM', 'FGA', 'DriveM', 'DriveA', 'DunkM', 'DunkA', 'FGM2',
                                  'FGA2', 'FGM3', 'FGA3', 'Pull3A', 'Pull3M'])
    gameID = 0
    for idx, grp in tqdm(events.groupby(['Season', 'DayNum', 'WTeamID', 'LTeamID'])):
        grp = grp.loc[grp['EventPlayerID'] != 0]
        gf = pd.DataFrame(columns=pdata.columns)
        gf['PlayerID'] = list(set(grp['EventPlayerID']))
        gf['GameID'] = gameID
        gf = gf.fillna(0)
        for p_idx, play in grp.groupby(['EventPlayerID']):
            if p_idx == 0:
                continue
            i = gf.loc[gf['PlayerID'] == p_idx].index[0]
            gf.loc[i, 'Ast'] = play.loc[play['EventType'] == 'assist'].shape[0]
            gf.loc[i, 'Blk'] = play.loc[play['EventType'] == 'block'].shape[0]
            gf.loc[i, 'Stl'] = play.loc[play['EventType'] == 'steal'].shape[0]
            gf.loc[i, 'Foul'] = play.loc[play['EventType'] == 'foul'].shape[0]
            gf.loc[i, 'TO'] = play.loc[play['EventType'] == 'turnover'].shape[0]
            gf.loc[i, 'Tech'] = play.loc[play['EventSubType'] == 'tech'].shape[0]
            gf.loc[i, 'OR'] = play.loc[play['EventSubType'] == 'off'].shape[0]
            gf.loc[i, 'DR'] = play.loc[play['EventSubType'] == 'def'].shape[0]
            
            s3 = play.loc[np.logical_or(play['EventType'] == 'made3',
                                         play['EventType'] == 'miss3')]
            s2 = play.loc[np.logical_or(play['EventType'] == 'made2',
                                         play['EventType'] == 'miss2')]
            s1 = play.loc[np.logical_or(play['EventType'] == 'made1',
                                         play['EventType'] == 'miss1')]
            if s3.shape[0] > 0:
                gf.loc[i, 'FGA3'] = s3.shape[0]
                gf.loc[i, 'FGM3'] = s3.loc[s3['EventType'] == 'made3'].shape[0]
                gf.loc[i, 'Pull3M'] = sum(np.logical_and(s3['EventType'] == 'made3',
                                                     s3['EventSubType'] == 'pullu'))
                gf.loc[i, 'Pull3A'] = s3.loc[s3['EventSubType'] == 'pullu'].shape[0]
            if s2.shape[0] > 0:
                gf.loc[i, 'FGA2'] = s2.shape[0]
                gf.loc[i, 'FGM2'] = s2.loc[s2['EventType'] == 'made2'].shape[0]
                gf.loc[i, 'DriveA'] = s2.loc[s2['EventSubType'] == 'drive'].shape[0]
                gf.loc[i, 'DunkA'] = s2.loc[s2['EventSubType'] == 'dunk'].shape[0]
                gf.loc[i, 'DriveM'] = sum(np.logical_and(s2['EventType'] == 'made2',
                                                     s2['EventSubType'] == 'drive'))
                gf.loc[i, 'DunkM'] = sum(np.logical_and(s2['EventType'] == 'made2',
                                                     s2['EventSubType'] == 'dunk'))
            gf.loc[i, 'FTA'] = s1.shape[0]
            gf.loc[i, 'FTM'] = s1.loc[s1['EventType'] == 'made1'].shape[0]
        pdata = pdata.append(gf, ignore_index=True)
        gameID += 1
    pdata = pdata.set_index(['GameID', 'PlayerID'])
    pdata['Pts'] = pdata['FGM3'] * 3 + pdata['FGM2'] * 2 + pdata['FTM']

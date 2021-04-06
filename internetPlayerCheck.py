#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 11:36:28 2021

@author: jeff
"""

from sportsreference.ncaab.roster import Player
from sportsreference.ncaab.teams import Teams, Team
from sportsreference.ncaab.boxscore import Boxscore, Boxscores
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from difflib import get_close_matches


seas_map = {'2010-2011': 2011, '2011-12': 2012,
                '2012-13': 2013, '2013-14': 2014,
                '2014-15': 2015, '2015-16': 2016,
                                        '2016-17': 2017, '2017-18': 2018,
                                        '2018-19': 2019, '2019-20': 2020,
                                        '2020-21': 2021, '2021-2022': 2022}

def getFromInternet(L, tms):
    ros = tms.roster
    L.append([p.dataframe for p in ros.players])
    
def mergeDF(L, df):
    ret = pd.DataFrame()
    for tm in df:
        for p in tm:
            ret = ret.append(p.reset_index(), ignore_index=True)
    L.append(ret)

master_p = []
for yr in tqdm(np.arange(2012, 2022)):
    with mp.Manager() as man:
        pdata = man.list()
        processes = []
        tms = Teams(yr)
        for f in tms:
            p = mp.Process(target=getFromInternet, args=(pdata, f,))
            processes.append(p)
            p.start()
        for proc in processes:
            proc.join()
        pdata = list(pdata)
    master_p.append(pdata)
    
play_df = pd.DataFrame()
tmp_p = []

with mp.Manager() as man:
    pd2 = man.list()
    processes = []
    for f in master_p:
        p = mp.Process(target=mergeDF, args=(pd2, f,))
        processes.append(p)
        p.start()
    for proc in processes:
        proc.join()
    tmp_p.append(list(pd2))
    
for yr in tmp_p:
    play_df = play_df.append(yr, ignore_index=True)
            
        
sdf = play_df.drop_duplicates()
sdf = sdf.dropna()
sdf = sdf.rename(columns={'level_0': 'Season'})
sdf = sdf.loc[sdf['Season'] != 'Career']
sdf['position'] = sdf['position'].map({'Center': 5, 'Forward': 3, 'Guard': 1})
#sdf = sdf.fillna(0)
sdf['height'] = sdf['height'].map({'5-10': 70, '5-11': 71, '5-2': 62, '5-5': 65,
                               '5-6': 66, '5-7': 67, '5-8': 68, '5-9': 69,
                               '6-0': 72, '6-1': 73, '6-10': 82, '6-11': 83,
                               '6-2': 74, '6-3': 75, '6-4': 76, '6-5': 77,
                               '6-6': 78, '6-7': 79, '6-8': 80, '6-9': 81,
                               '7-0': 84, '7-1': 85, '7-2': 86, '7-3': 87,
                               '7-4': 88, '7-6': 89, '7-7': 90})
sdf = sdf.dropna()
sdf['Season'] = sdf['Season'].map(seas_map)
sdf = sdf.dropna()
sdf['Season'] = sdf['Season'].astype(int)

pdf = pd.read_csv('./data/InternetPlayerNames.csv')
for idx, row in pdf.iterrows():
    pdf.loc[idx, 'PlayerName'] = row['PlayerName'].lower()
tid_conv = pd.read_csv('./data/InternetTeamNameToTID.csv')
abbs = {}

sdf['PlayerID'] = -1
cnt = 0
for idx, grp in tqdm(sdf.groupby(['player_id'])):
    sdf.loc[grp.index, 'PlayerID'] = cnt
    nme = idx.split('-')
    rnme = ''
    for n in nme[:-1]:
        rnme += n + ' '
    matches = get_close_matches(rnme[:-1], pdf['PlayerName'], cutoff=.8)
    if len(matches) == 1:
        tabb = list(set(grp['team_abbreviation']))
        if len(tabb) == 1:
            if tabb[0] not in abbs:
                abbs[tabb[0]] = pdf.loc[pdf['PlayerName'] == matches[0], 'TID'].values[0]
    cnt += 1
    
sdf['TID'] = -1
for idx, grp in sdf.groupby(['team_abbreviation']):
    if idx in abbs:
        sdf.loc[grp.index, 'TID'] = abbs[idx]

re_abbs = {value:key for key, value in abbs.items()}
for idx in re_abbs:
    tid_conv.loc[tid_conv['TID'] == idx, 'ITNme'] = re_abbs[idx]
    
for idx, row in tqdm(sdf.iterrows()):
    sdf.loc[idx, 'TID'] = tid_conv.loc[tid_conv['ITNme'] == row['team_abbreviation'], 'TID'].values[0]
    
sdf = sdf.set_index(['Season', 'PlayerID', 'TID'])
sdf = sdf.drop(columns=['conference', 'player_id', 'team_abbreviation'])
sdf.columns = ['Ast%', 'Ast', 'Blk%', 'Blk', 'BPM', 'DBPM', 'DR%', 'DR', 'DWS',
               'eFG%', 'FGA', 'FG%', 'FGM', 'FT/A', 'FTA', 'FT%', 'FTM', 'GP', 'GS',
               'Height', 'Mins', 'OBPM', 'OR%', 'OR', 'OWS', 'PF', 'PER', 'Pts',
               'PtsProd', 'Pos', 'Stl%', 'Stl', '3/2Rate', 'FGA3', 'FG3%', 'FGM3',
               'R%', 'R', 'TS%', 'TO%', 'TO', 'FGA2', 'FG2%', 'FGM2', 'Usage%',
               'Weight', 'WS', 'WSPer40']

sdf.to_csv('./data/InternetPlayerData.csv')
    

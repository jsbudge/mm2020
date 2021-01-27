#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:31:12 2021

@author: jeff
"""
from sportsreference.ncaab.roster import Player
from sportsreference.ncaab.teams import Teams, Team
import pandas as pd
from tqdm import tqdm
import statslib as st
import eventlib as ev
import numpy as np
from difflib import get_close_matches

files = st.getFiles()
pdf = ev.getTeamRosters(files)
#Grab the data from sportsreference
seas_map = {'2014-15': 2015, '2015-16': 2016,
                                   '2016-17': 2017, '2017-18': 2018,
                                   '2018-19': 2019, '2019-20': 2020}
play_df = pd.DataFrame()
for season in [2015, 2016, 2017, 2018, 2019, 2020]:
    tm = Teams(season)
    for team in tqdm(tm):
        ros = team.roster
        for play in ros.players:
            if play_df.shape[0] > 0:
                if play.player_id not in play_df['player_id']:
                    t = play.dataframe
                    t = t.reset_index().fillna(0)
                    t['level_0'] = t['level_0'].map(seas_map)
                    t['PlayerName'] = play.name
                    t['TNme'] = team.name
                    play_df = play_df.append(t.loc[t['level_0'] == season], ignore_index=True)
                else:
                    play_df.loc[np.logical_and(play_df['player_id'] == play.player_id,
                                               play_df['level_0'] == season), 'TNme'] = team.name
            else:
                t = play.dataframe
                t = t.reset_index().fillna(0)
                t['level_0'] = t['level_0'].map(seas_map)
                t['PlayerName'] = play.name
                t['TNme'] = team.name
                play_df = play_df.append(t.loc[t['level_0'] == season], ignore_index=True)

#%%
#Rearrange things to my liking
sdf = play_df.copy()
sdf = sdf.drop_duplicates()
sdf = sdf.rename(columns={'level_0': 'Season'})
sdf['position'] = sdf['position'].map({'Center': 5, 'Forward': 3, 'Guard': 1})
sdf['height'] = sdf['height'].map({'5-10': 70, '5-11': 71, '5-2': 62, '5-5': 65,
                                   '5-6': 66, '5-7': 67, '5-8': 68, '5-9': 69,
                                   '6-0': 72, '6-1': 73, '6-10': 82, '6-11': 83,
                                   '6-2': 74, '6-3': 75, '6-4': 76, '6-5': 77,
                                   '6-6': 78, '6-7': 79, '6-8': 80, '6-9': 81,
                                   '7-0': 84, '7-1': 85, '7-2': 86, '7-3': 87,
                                   '7-4': 88, '7-6': 89, '7-7': 90})
sdf = sdf.dropna()
sdf['Season'] = sdf['Season'].astype(int)
sdf['PlayerID'] = -1
sdf['TID'] = -1
sdf['FullName'] = '0'
curr_tid = 1101
close_ones = []
for idx, grp in tqdm(sdf.groupby(['TNme', 'PlayerName'])):
    link = pdf.loc[pdf['FullName'] == idx[1]]
    try:
        if len(link) > 1:
            continue
            link = link.loc[abs(link['TeamID'] - curr_tid) == abs(link['TeamID'] - curr_tid).min()]
        ids = [link.index.values[0], link['TeamID'].values[0], link['FullName'].values[0]]
        curr_tid = link['TeamID'].values[0]
    except:
        continue
        # try:
        #     poss_match = get_close_matches(idx[1], pdf['FullName'])[0]
        #     link = pdf.loc[pdf['FullName'] == poss_match]
        #     ids = [link.index.values[0], link['TeamID'].values[0], link['FullName'].values[0]]
        #     close_ones.append(ids)
        #     curr_tid = link['TeamID'].values[0]
        # except:
        #     print('Unknown player ' + idx[1])
        #     ids = [-1, -1, '0']
    sdf.loc[grp.index, ['PlayerID', 'TID', 'FullName']] = ids

abbs = {}
for idx, grp in sdf.groupby(['TNme']):
    abbs[idx] = grp.loc[grp['TID'] != -1, 'TID'].mode().values[0]
for idx, row in sdf.iterrows():
    sdf.loc[idx, 'TID'] = abbs[row['TNme']]
    if row['PlayerID'] == -1:
        poss_matches = get_close_matches(row['PlayerName'], pdf.loc[pdf['TeamID'] == abbs[row['TNme']],
                                                                    'FullName'])
        if len(poss_matches) > 0:
            match = pdf.loc[np.logical_and(pdf['FullName'] == poss_matches[0],
                                           pdf['TeamID'] == abbs[row['TNme']])]
            sdf.loc[idx, ['FullName', 'PlayerID']] = [match['FullName'].values[0],
                                                      match.index.values[0]]
            
sdf = sdf.set_index(['Season', 'PlayerID', 'TID'])
sdf = sdf.drop(columns=['conference', 'player_id', 'team_abbreviation', 
                        'PlayerName', 'FullName', 'TNme'])
sdf.columns = ['Ast%', 'Ast', 'Blk%', 'Blk', 'BPM', 'DBPM', 'DR%', 'DR', 'DWS',
               'eFG%', 'FGA', 'FG%', 'FGM', 'FT/A', 'FTA', 'FT%', 'FTM', 'GP', 'GS',
               'Height', 'Mins', 'OBPM', 'OR%', 'OR', 'OWS', 'PF', 'PER', 'Pts',
               'PtsProd', 'Pos', 'Stl%', 'Stl', '3/2Rate', 'FGA3', 'FG3%', 'FGM3',
               'R%', 'R', 'TS%', 'TO%', 'TO', 'FGA2', 'FG2%', 'FGM2', 'Usage%',
               'Weight', 'WS', 'WSPer40']

sdf.to_csv('./data/InternetPlayerData.csv')

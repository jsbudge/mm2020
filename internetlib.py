#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:31:12 2021

@author: jeff

Generates player data from sportsreference.com and matches it to
the event data in MEvents{}.csv. Saves the results out to another csv,
because this takes for-ev-er to run.
"""
from sportsreference.ncaab.roster import Player
from sportsreference.ncaab.teams import Teams, Team
from sportsreference.ncaab.boxscore import Boxscore, Boxscores
import pandas as pd
from tqdm import tqdm
import statslib as st
import eventlib as ev
import numpy as np
from difflib import get_close_matches

seas_map = {'2011-12': 2012,
                '2012-13': 2013, '2013-14': 2014,
                '2014-15': 2015, '2015-16': 2016,
                                        '2016-17': 2017, '2017-18': 2018,
                                        '2018-19': 2019, '2019-20': 2020,
                                        '2020-21': 2021, '2021-2022': 2022}

def getPlayerData(seasons=None, add_to_existing=True):
    pdf = ev.getTeamRosters()
    #Grab the data from sportsreference
    play_df = pd.DataFrame()
    if seasons is None:
        seasons = range(2012, 2022)
    for season in seasons:
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
    for idx, grp in tqdm(sdf.groupby(['TNme', 'PlayerName'])):
        link = pdf.loc[pdf['FullName'] == idx[1]]
        if len(link) == 1:
            ids = [link.index.values[0], link['TeamID'].values[0], link['FullName'].values[0]]
            sdf.loc[grp.index, ['PlayerID', 'TID', 'FullName']] = ids
    
    abbs = {}
    for idx, grp in sdf.groupby(['TNme']):
        abbs[idx] = grp.loc[grp['TID'] != -1, 'TID'].mode().values[0]
        sdf.loc[grp.index, 'TID'] = abbs[idx]
    for idx, grp in tqdm(sdf.loc[sdf['PlayerID'] == -1].groupby(['TID', 'PlayerName'])):
        ids = ['0', -1]
        match = pdf.loc[np.logical_and(pdf['FullName'] == idx[1],
                                       pdf['TeamID'] == idx[0])]
        if len(match) > 0:
            if len(match) == 1:
                ids = [match['FullName'].values[0], match.index.values[0]]
            else:
                ids = []
                n_blanks = int(grp.shape[0] / match.shape[0])
                n_overflow = grp.shape[0] % match.shape[0]
                for m_idx, m_row in match.iterrows():
                    for mm in range(n_blanks):
                        ids.append([m_row['FullName'], m_idx])
                for mm in range(n_overflow):
                    ids.append([m_row['FullName'], m_idx])
        else:
            lastname = idx[1].split(' ')[-1]
            if lastname in ['Jr.', 'Jr', 'III', 'II', 'V', 'IV']:
                lastname = idx[1].split(' ')[-2]
            poss_matches = get_close_matches(lastname, 
                                                     pdf.loc[pdf['TeamID'] == idx[0],
                                                             'LastName'], cutoff=.7)
            if len(poss_matches) == 1:
                match = pdf.loc[np.logical_and(pdf['LastName'] == poss_matches[0],
                                           pdf['TeamID'] == idx[0])]
                ids = [match['FullName'].values[0], match.index.values[0]]
            elif len(poss_matches) > 1:
                poss_matches = get_close_matches(idx[1], 
                                                 pdf.loc[pdf['TeamID'] == idx[0],
                                                 'FullName'])
                if len(poss_matches) == 1:
                    match = pdf.loc[np.logical_and(pdf['FullName'] == poss_matches[0],
                                                   pdf['TeamID'] == idx[0])]
                    if len(match) == 1:
                        ids = [match['FullName'].values[0], match.index.values[0]]
                elif len(poss_matches) > 1:
                    if poss_matches[0] == poss_matches[1]:
                        match = pdf.loc[np.logical_and(pdf['FullName'] == poss_matches[0],
                                                       pdf['TeamID'] == idx[0])]
                        ids = []
                        n_blanks = int(grp.shape[0] / match.shape[0])
                        n_overflow = grp.shape[0] % match.shape[0]
                        for m_idx, m_row in match.iterrows():
                            for mm in range(n_blanks):
                                ids.append([m_row['FullName'], m_idx])
                        for mm in range(n_overflow):
                            ids.append([m_row['FullName'], m_idx])
                    else:
                        match = pdf.loc[np.logical_and(pdf['FullName'] == poss_matches[0],
                                                       pdf['TeamID'] == idx[0])]
                        if len(match) == 1:
                            ids = [match['FullName'].values[0], match.index.values[0]]
                else:
                    #Some of the names are switched
                    rev_name = pdf.loc[pdf['TeamID'] == idx[0], 'LastName'] + \
                        ' ' + pdf.loc[pdf['TeamID'] == idx[0], 'FirstName']
                    poss_matches = get_close_matches(idx[1], rev_name)
                    if len(poss_matches) > 0:
                        for m in poss_matches:
                            match = pdf.loc[rev_name.index[rev_name == m]]
                            if len(match) > 0:
                                ids = [match['FullName'].values[0], match.index.values[0]]
        sdf.loc[grp.index, ['FullName', 'PlayerID']] = ids
    
    unmatched = []
    for idx, row in pdf.iterrows():
        if idx not in sdf['PlayerID'].values:
            unmatched.append(row)
    unmatched = pd.DataFrame(unmatched)
    
    nid = -2
    for idx, grp in sdf.loc[sdf['PlayerID'] == -1].groupby(['PlayerName', 'TID']):
        sdf.loc[grp.index, 'PlayerID'] = nid
        nid -= 1
    
    sdf = sdf.set_index(['Season', 'PlayerID', 'TID'])
    sdf = sdf.drop(columns=['conference', 'player_id', 'team_abbreviation', 
                            'PlayerName', 'FullName', 'TNme'])
    sdf.columns = ['Ast%', 'Ast', 'Blk%', 'Blk', 'BPM', 'DBPM', 'DR%', 'DR', 'DWS',
                   'eFG%', 'FGA', 'FG%', 'FGM', 'FT/A', 'FTA', 'FT%', 'FTM', 'GP', 'GS',
                   'Height', 'Mins', 'OBPM', 'OR%', 'OR', 'OWS', 'PF', 'PER', 'Pts',
                   'PtsProd', 'Pos', 'Stl%', 'Stl', '3/2Rate', 'FGA3', 'FG3%', 'FGM3',
                   'R%', 'R', 'TS%', 'TO%', 'TO', 'FGA2', 'FG2%', 'FGM2', 'Usage%',
                   'Weight', 'WS', 'WSPer40']
    if add_to_existing:
        ipd = pd.read_csv('./data/InternetPlayerData.csv').set_index(['Season', 'PlayerID', 'TID']).sort_index()
        sdf = sdf.append(ipd)
    else:
        sdf.to_csv('./data/InternetPlayerData.csv')
        
def getTeamSeasonalData(seasons=None, add_to_existing=True):
    #Grab the data from sportsreference
    play_df = pd.DataFrame()
    if seasons is None:
        seasons = np.arange(2012, 2022)
    for season in seasons:
        tm = Teams(season)
        for team in tqdm(tm):
            try:
                df = team.dataframe
                df['Season'] = season
                play_df = play_df.append(df)
            except:
                print(team.name)
    play_df['TID'] = 0
    idx = 1
    for tid in list(set(play_df['abbreviation'])):
        play_df.loc[play_df['abbreviation'] == tid, 'TID'] = idx
        idx += 1
    play_df = play_df.set_index(['Season', 'TID'])
            
    if add_to_existing:
        ipd = pd.read_csv('./data/InternetTeamSeasonData.csv').set_index(['Season', 'TID']).sort_index()
        play_df = play_df.append(ipd)
    play_df.to_csv('./data/InternetPlayerData.csv')
    
def getGameData(seasons=None, add_to_existing=False):
    play_df = pd.DataFrame()
    if season is None:
        seasons = np.arange(2012, 2022)
    for season in seasons:
        gms = Boxscores(datetime(season-1, 8, 1), datetime(season, 4, 30))
        for gm in tqdm(gms.games):
            for g in gms.games[gm]:
                play_df = play_df.append(Boxscore(g['boxscore']).dataframe)
    new_cols = []
    for col in play_df.columns:
        nc = col.replace('_', '')
        nc = nc.replace('home', 'T_')
        nc = nc.replace('away', 'O_')
        nc = nc.replace('fieldgoal', 'FG')
        nc = nc.replace('rating', 'Rtg')
        nc = nc.replace('twopoint', '2Pt')
        nc = nc.replace('percentage', '%')
        nc = nc.replace('threepoint', '3Pt')
        nc = nc.replace('freethrow', 'FT')
        nc = nc.replace('defensive', 'D')
        nc = nc.replace('offensive', 'O')
        nc = nc.replace('ranking', 'Rank')
        nc = nc.replace('effective', 'e')
        nc = nc.replace('rebound', 'R')
        nc = nc.replace('attempt', 'A')
        nc = nc.replace('block', 'Blk')
        nc = nc.replace('assist', 'Ast')
        nc = nc.replace('steal', 'Stl')
        nc = nc.replace('turnover', 'TO')
        nc = nc.replace('trueshooting', 'TS')
        nc = nc.replace('total', '')
        nc = nc.replace('points', 'Score')
        nc = nc.replace('personalfouls', 'PF')
        nc = nc.replace('rate', 'Rate')
        nc = nc.replace('pace', 'Pace')
        nc = nc[:-1] if nc[-1] == 's' else nc
        new_cols.append(nc)
    play_df.columns = new_cols
    return play_df

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
from tqdm import tqdm

def getRosters(files, season):
    evp = pd.read_csv(files['MEvents{}'.format(season)]).drop(columns=['Season']).set_index(['DayNum', 'WTeamID', 'LTeamID', 'EventPlayerID']).fillna(0)
    evp = evp.loc[evp.index.get_level_values(3) != 0]
    evp['Pivot'] = 1
    ev = evp.pivot_table(index=evp.index, columns=['EventType', 'EventSubType'], values='Pivot', aggfunc=sum).fillna(0)
    ev.index = pd.MultiIndex.from_tuples(ev.index)
    evst = pd.DataFrame(index=ev.index)
    players = pd.read_csv(files['MPlayers'])
    gameID = 0
    evst['Ast'] = ev['assist']
    evst['Blk'] = ev['block']
    evst['Foul'] = ev['foul'].sum(axis=1)
    evst['TO'] = ev['turnover'].sum(axis=1)
    evst['Stl'] = ev['steal']
    evst['FGM3'] = ev['made3'].sum(axis=1)
    evst['FGA3'] = ev['made3'].sum(axis=1) + ev['miss3'].sum(axis=1)
    evst['FGM2'] = ev['made2'].sum(axis=1)
    evst['FGA2'] = ev['made2'].sum(axis=1) + ev['miss2'].sum(axis=1)
    evst['FTM'] = ev['made1'].sum(axis=1)
    evst['FTA'] = ev['made1'].sum(axis=1) + ev['miss1'].sum(axis=1)
    evst['FGA'] = evst['FGA2'] + evst['FGA3']
    evst['FGM'] = evst['FGM2'] + evst['FGM3']
    evst['OR'] = ev['reb', 'off']
    evst['DR'] = ev['reb', 'def']
    evst['LayM'] = ev['made2', 'lay']
    evst['LayA'] = ev['made2', 'lay'] + ev['miss2', 'lay']
    evst['DunkM'] = ev['made2', 'dunk']
    evst['DunkA'] = ev['made2', 'dunk'] + ev['miss2', 'dunk']
    evst['Pts'] = evst['FGM3'] * 3 + evst['FGM2'] * 2 + evst['FTM']
    evst['Mins'] = 0
    poss_breaks = np.array([0, 1200, 2400, 2700, 3000, 3300, 3600, 3900])
    for idx, grp in tqdm(evp.groupby(['DayNum', 'WTeamID', 'LTeamID', 'EventPlayerID'])):
        mins = 0; lastSub = 0; subin = True
        grp = grp.loc[grp['EventType'] == 'sub']
        if grp.shape[0] == 0:
            continue
        for i, row in grp.iterrows():
            if row['EventSubType'] == 'in':
                lastSub = row['ElapsedSeconds']
                if subin:
                    br = poss_breaks - lastSub
                    br[br < 0] += 100000
                    mins += poss_breaks[br == br.min()] - lastSub
                subin = True
            elif row['EventSubType'] == 'out':
                mins += row['ElapsedSeconds'] - lastSub
                subin = False
        if subin:
            br = poss_breaks - lastSub
            br[br < 0] += 100000
            mins += poss_breaks[br == br.min()] - lastSub
        evst.loc[idx, 'Mins'] = np.round(mins / 60, 2)
    evst.index = evst.index.set_names(['DayNum', 'WTeamID', 'LTeamID', 'PlayerID'])
    return evst

def getAdvStats(df):
    wdf = pd.DataFrame(index=df.index)
    mins = df['Mins']
    for col in df.columns:
        if col != 'Mins':
            wdf[col + 'Per'] = df[col] / mins
    wdf['FG%'] = df['FGM'] / df['FGA']
    wdf['eFG%'] = (df['FGM'] + .5 * df['FGM3']) / df['FGA']
    wdf['TS%'] = df['Pts'] / (2 * (df['FGA'] + .44 * df['FTA']))
    wdf['Econ'] = df['Ast'] + df['Stl'] - df['TO']
    wdf['Smart%'] = (df['FGA3'] + df['LayA'] + df['DunkA']) / df['FGA']
    wdf['FT/A'] = df['FTA'] / df['FGA']
    wdf['FT%'] = df['FTM'] / df['FTA']
    return wdf.fillna(0)

def getPlayerSeasonStats(df):
    return df.groupby(['PlayerID']).mean()

def loadPlayerNames(files):
    df = pd.read_csv(files['MPlayers'])
    df['FullName'] = df['FirstName'] + ' ' + df['LastName']
    ret = {}
    for idx, row in df.iterrows():
        ret[row['PlayerID']] = row['FullName']
        ret[row['FullName']] = row['PlayerID']
    return ret
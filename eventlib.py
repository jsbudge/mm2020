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
    ev = pd.read_csv(files['MEvents{}'.format(season)]).drop(columns=['Season']).set_index(['DayNum', 'WTeamID', 'LTeamID', 'EventPlayerID']).fillna(0)
    mins = ev.groupby(['DayNum', 'WTeamID', 'LTeamID', 'EventPlayerID']).apply(lambda x: sum(x.loc[x['EventSubType'] == 'out', 'ElapsedSeconds']) - sum(x.loc[x['EventSubType'] == 'in', 'ElapsedSeconds']))
    mins.loc[mins < 0] += 2400
    ev['Pivot'] = 1
    ev = ev.pivot_table(index=ev.index, columns=['EventType', 'EventSubType'], values='Pivot', aggfunc=sum).fillna(0)
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
    evst['DriveM'] = ev['made2', 'lay']
    evst['DriveA'] = ev['made2', 'lay'] + ev['miss2', 'lay']
    evst['DunkM'] = ev['made2', 'dunk']
    evst['DunkA'] = ev['made2', 'dunk'] + ev['miss2', 'dunk']
    evst['Pts'] = evst['FGM3'] * 3 + evst['FGM2'] * 2 + evst['FTM']
    evst['Mins'] = mins / 60
    return evst

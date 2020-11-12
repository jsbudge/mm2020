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


poss_breaks = np.array([1200, 2400, 2700, 3000, 3300, 3600, 3900, 4200])

'''
Gets stats for every player in every game.
Note that the player IDs do not match the MPlayerID file, for some reason.
THe stats themselves are correct for the ID, though, cross-checked
against sports-reference.com.
'''
def getRosters(files, season):
    evp = pd.read_csv(files['MEvents{}'.format(season)]).drop(columns=['Season']).set_index(['DayNum', 'WTeamID', 'LTeamID', 'EventPlayerID']).fillna(0)
    evp['GameID'] = evp.groupby(['DayNum', 'WTeamID', 'LTeamID']).ngroup()
    evp = evp.reset_index().set_index(['GameID', 'WTeamID', 'LTeamID', 'EventPlayerID'])
    evp = evp.loc[evp.index.get_level_values(3) != 0]
    evp['Pivot'] = 1
    ev = evp.pivot_table(index=evp.index, columns=['EventType', 'EventSubType'], values='Pivot', aggfunc=sum).fillna(0)
    ev.index = pd.MultiIndex.from_tuples(ev.index)
    evst = pd.DataFrame(index=ev.index)
    evst.index = evst.index.set_names(['GameID', 'WTeamID', 'LTeamID', 'PlayerID'])
    evst = evst.reset_index().set_index(['GameID', 'PlayerID'])
    teams = evst.groupby(['GameID']).mean()
    evst['Mins'] = 0
    
    #Load things into memory, which speeds up the minutes calculations but...uses more memory.
    maxp = evp.groupby(['GameID']).apply(lambda x: poss_breaks[:sum(poss_breaks <= x['ElapsedSeconds'].max())+1])
    subins = evp.loc[evp['EventSubType'] == 'in'].groupby(['GameID', 'EventPlayerID']).apply(lambda x: x['ElapsedSeconds'].values)
    subouts = evp.loc[evp['EventSubType'] == 'out'].groupby(['GameID', 'EventPlayerID']).apply(lambda x: x['ElapsedSeconds'].values)
    for idx, grp in tqdm(evp.groupby(['GameID', 'EventPlayerID'])):
        pb = maxp.loc[idx[0]] #poss_breaks[:sum(poss_breaks <= grp['ElapsedSeconds'].max())+1]
        try:
            subout = subouts.loc[idx] #grp.loc[grp['EventSubType'] == 'out', 'ElapsedSeconds'].values
        except:
            subout = []
        try:
            subin = subins.loc[idx] #grp.loc[grp['EventSubType'] == 'in', 'ElapsedSeconds'].values
        except:
            subin = []
        if len(subout) == 0 and len(subin) == 0:
            mins = pb[-1]
        else:
            subs = [('si', n) for n in subin] + [('so', n) for n in subout]
            subs.sort(key = lambda x: x[1])
            mins = 0; so = 0; si = 0
            ing = False if subs[0][0] == 'si' else True
            for (t, m) in subs:
                if t == 'si':
                    if not ing:
                        si = m
                        ing = True
                if t == 'so':
                    if ing:
                        so = m
                        mins += so - si
                        ing = False
            if ing:
                mins += pb[-1] - si
        evst.loc[idx, 'Mins'] = np.round(mins / 60, 2)
    evst['Ast'] = ev['assist'].values
    evst['Blk'] = ev['block'].values
    evst['Foul'] = ev['foul'].sum(axis=1).values
    evst['TO'] = ev['turnover'].sum(axis=1).values
    evst['Stl'] = ev['steal'].values
    evst['FGM3'] = ev['made3'].sum(axis=1).values
    evst['FGA3'] = ev['made3'].sum(axis=1).values + ev['miss3'].sum(axis=1).values
    evst['FGM2'] = ev['made2'].sum(axis=1).values
    evst['FGA2'] = ev['made2'].sum(axis=1).values + ev['miss2'].sum(axis=1).values
    evst['FTM'] = ev['made1'].sum(axis=1).values
    evst['FTA'] = ev['made1'].sum(axis=1).values + ev['miss1'].sum(axis=1).values
    evst['FGA'] = evst['FGA2'].values + evst['FGA3'].values
    evst['FGM'] = evst['FGM2'].values + evst['FGM3'].values
    evst['OR'] = ev['reb', 'off'].values
    evst['DR'] = ev['reb', 'def'].values
    evst['LayM'] = ev['made2', 'lay'].values
    evst['LayA'] = ev['made2', 'lay'].values + ev['miss2', 'lay'].values
    evst['DunkM'] = ev['made2', 'dunk'].values
    evst['DunkA'] = ev['made2', 'dunk'].values + ev['miss2', 'dunk'].values
    evst['Pts'] = evst['FGM3'].values * 3 + evst['FGM2'].values * 2 + evst['FTM'].values
    return teams, evst.drop(columns=['WTeamID', 'LTeamID']).fillna(0)

def getEvents(files, season):
    return pd.read_csv(files['MEvents{}'.format(season)]).drop(columns=['Season']).set_index(['DayNum', 'WTeamID', 'LTeamID', 'EventPlayerID']).fillna(0)

def getSingleGameLineups(gid, files, season):
    evp = pd.read_csv(files['MEvents{}'.format(season)]).drop(columns=['Season']).set_index(['DayNum', 'WTeamID', 'LTeamID', 'EventPlayerID']).fillna(0)
    evp['GameID'] = evp.groupby(['DayNum', 'WTeamID', 'LTeamID']).ngroup()
    evp = evp.reset_index().set_index(['GameID', 'EventPlayerID'])
    #evp = evp.loc[evp.index.get_level_values(1) != 0]
    ev = evp.loc(axis=0)[gid, :]
    gl = poss_breaks[poss_breaks >= ev['ElapsedSeconds'].max()].min()
    pb = poss_breaks[poss_breaks <= gl]
    nms = ('W', 'L')
    ids = (ev['WTeamID'].values[0], ev['LTeamID'].values[0])
    game = (pd.DataFrame(index=np.arange(gl)),
            pd.DataFrame(index=np.arange(gl)))
    for t in range(2):
        ts = 0
        wev = ev.loc[ev['EventTeamID'] == ids[t], ['EventType', 'EventSubType', 'ElapsedSeconds']]
        players = np.sort(list(set(wev.index.get_level_values(1))))
        inglist = np.zeros((gl,))
        for p in players:
            psub = wev.loc(axis=0)[:, p]
            try:
                subout = psub.loc[psub['EventSubType'] == 'out', 'ElapsedSeconds'].values
            except:
                subout = []
            try:
                subin = psub.loc[psub['EventSubType'] == 'in', 'ElapsedSeconds'].values
            except:
                subin = []
            if len(subout) == 0 and len(subin) == 0:
                inglist += 1
            else:
                subs = [('si', n) for n in subin] + [('so', n) for n in subout]
                subs.sort(key = lambda x: x[1])
                mins = 0; so = 0; si = 0
                ing = False if subs[0][0] == 'si' else True
                for n in range(gl):
                    if n in subin:
                        ing = True
                        si += 1 if si + 1 < len(subin) else 0
                    if n in subout:
                        ing = False
                        so += 1 if so + 1 < len(subout) else 0
                    if n in pb:
                        if subout[so] > n and subin[si] > subout[so]:
                            ing = True
                        else:
                            ing = False
                    inglist[n] = ing
            game[t]['P{}'.format(p)] = inglist
        nump = np.sum(game[t], axis=1)
    return 0

def getAdvStats(df):
    wdf = pd.DataFrame(index=df.index)
    mins = df['Mins']
    mins[mins < 5] = 5 #Removes exploding values
    for col in df.columns:
        if col != 'Mins':
            wdf[col + 'Per36'] = df[col]  * 36 / mins.values
    wdf['FG%'] = df['FGM'] / df['FGA']
    wdf['eFG%'] = (df['FGM'] + .5 * df['FGM3']) / df['FGA']
    wdf['TS%'] = df['Pts'] / (2 * (df['FGA'] + .44 * df['FTA']))
    wdf['Econ'] = df['Ast'] + df['Stl'] - df['TO']
    wdf['Smart%'] = (df['FGA3'] + df['LayA'] + df['DunkA']) / df['FGA']
    wdf['FT/A'] = df['FTA'] / df['FGA']
    wdf['FT%'] = df['FTM'] / df['FTA']
    return wdf.dropna()

def getPlayerSeasonStats(df):
    return df.groupby(['PlayerID']).mean()

def getTeamRosters(files):
    df = pd.read_csv(files['MPlayers'])
    df['FullName'] = df['FirstName'] + ' ' + df['LastName']
    return df

'''
THESE ARE NOT CORRECT.
'''
def loadPlayerNames(files):
    df = pd.read_csv(files['MPlayers'])
    df['FullName'] = df['FirstName'] + ' ' + df['LastName']
    ret = {}
    for idx, row in df.iterrows():
        ret[row['PlayerID']] = row['FullName']
        if row['FullName'] in ret:
            try:
                ret[row['FullName']] = ret[row['FullName']] + [row['PlayerID']]
            except:
                ret[row['FullName']] = [ret[row['FullName']], row['PlayerID']]
        else:
            ret[row['FullName']] = row['PlayerID']
    return ret
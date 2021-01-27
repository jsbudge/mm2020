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
from sklearn.preprocessing import OrdinalEncoder
from dataclasses import dataclass
from itertools import groupby

def all_equal(i, q):
    return np.all([ii in q for ii in i])

poss_breaks = np.array([0, 1200, 2400, 2700, 3000, 3300, 3600, 3900, 4200])

'''
Gets stats for every player in every game.
'''
def getRosters(files, season):
    evp = pd.read_csv(files['MEvents{}'.format(season)]).drop(columns=['Season']).set_index(['DayNum', 'WTeamID', 'LTeamID', 'EventPlayerID']).fillna(0)
    evp['GameID'] = evp.groupby(['DayNum', 'WTeamID', 'LTeamID']).ngroup()
    evp = evp.reset_index().set_index(['GameID', 'DayNum', 'WTeamID', 'LTeamID', 'EventPlayerID'])
    evp = evp.loc[evp.index.get_level_values(4) != 0]
    evp['Pivot'] = 1
    ev = evp.pivot_table(index=evp.index, columns=['EventType', 'EventSubType'], values='Pivot', aggfunc=sum).fillna(0)
    ev.index = pd.MultiIndex.from_tuples(ev.index)
    evst = pd.DataFrame(index=ev.index)
    evst.index = evst.index.set_names(['GameID', 'DayNum', 'WTeamID', 'LTeamID', 'PlayerID'])
    evst = evst.reset_index().set_index(['GameID', 'PlayerID'])
    teams = evst.groupby(['GameID']).mean()
    evst['Mins'] = 0
    
    #Load things into memory, which speeds up the minutes calculations but...uses more memory.
    maxp = evp.groupby(['GameID']).apply(lambda x: poss_breaks[:sum(poss_breaks <= x['ElapsedSeconds'].max())+1])
    subins = evp.loc[evp['EventSubType'] == 'in'].groupby(['GameID', 'EventPlayerID']).apply(lambda x: x['ElapsedSeconds'].values - .5)
    subouts = evp.loc[evp['EventSubType'] == 'out'].groupby(['GameID', 'EventPlayerID']).apply(lambda x: x['ElapsedSeconds'].values + .5)
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
        breaks = np.sort(list(pb) + list(subout) + list(subin))
        nev, secs = np.histogram(grp['ElapsedSeconds'], breaks)
        mins = sum([secs[n+1] - secs[n] for n in range(len(nev)) if nev[n] > 0])
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
    evp = pd.read_csv(files['MEvents{}'.format(season)]).drop(columns=['Season']).set_index(['DayNum', 'WTeamID', 'LTeamID', 'EventPlayerID'])
    evp['GameID'] = evp.groupby(['DayNum', 'WTeamID', 'LTeamID']).ngroup()
    evp = evp.reset_index().set_index(['GameID', 'EventID'])
    ordenc = OrdinalEncoder()
    ev = evp.loc(axis=0)[gid, :].copy()
    wid = ev['WTeamID'].values[0]; lid = ev['LTeamID'].values[0]
    ev['EventTypeID'] = ordenc.fit_transform(np.array([row['EventType'] + str(row['EventSubType']) for idx, row in ev.iterrows()]).reshape((-1, 1))).astype(int)
    ev = ev.drop(columns=['DayNum', 'X', 'Y', 'Area',
                          'WTeamID', 'LTeamID'])
    ev = ev.loc[ev['EventPlayerID'] != 0]
    mxepsec = ev['ElapsedSeconds'].max()
    ov = [0, 1200, 2400]
    while ov[-1] < mxepsec:
        ov = ov + [ov[-1] + 300]
    for idx, row in ev.iterrows():
        if row['EventTeamID'] == wid:
            if 'made1' in row['EventType']:
                ev.loc[idx, 'WCurrentScore'] = 1
            elif 'made2' in row['EventType']:
                ev.loc[idx, 'WCurrentScore'] = 2
            elif 'made3' in row['EventType']:
                ev.loc[idx, 'WCurrentScore'] = 3
            else:
                ev.loc[idx, 'WCurrentScore'] = 0
        else:
            if 'made1' in row['EventType']:
                ev.loc[idx, 'LCurrentScore'] = 1
            elif 'made2' in row['EventType']:
                ev.loc[idx, 'LCurrentScore'] = 2
            elif 'made3' in row['EventType']:
                ev.loc[idx, 'LCurrentScore'] = 3
            else:
                ev.loc[idx, 'LCurrentScore'] = 0
    ev['WCurrentScore'] = np.cumsum(ev['WCurrentScore'])
    ev['LCurrentScore'] = np.cumsum(ev['LCurrentScore'])
    lns = (pd.DataFrame(index=np.arange(ov[-1]), columns=list(set(ev.loc[ev['EventTeamID'] == wid,
                                                                         'EventPlayerID'].values)),
                         data=False),
           pd.DataFrame(index=np.arange(ov[-1]), columns=list(set(ev.loc[ev['EventTeamID'] == lid,
                                                                         'EventPlayerID'].values)),
                         data=False))
    for pid, p in ev.groupby(['EventPlayerID']):
        if pid != 0:
            poss_subs = np.sort(ov + \
                list(p.loc[p['EventSubType'] == 'in', 'ElapsedSeconds'] - .5) + \
                    list(p.loc[p['EventSubType'] == 'out', 'ElapsedSeconds'] + .5))
            nev, secs = np.histogram(p['ElapsedSeconds'], poss_subs)
            for n in range(len(nev)):
                if nev[n] > 0:
                    if pid in lns[0].columns:
                        lns[0].loc[np.logical_and(lns[0].index > secs[n],
                                                  lns[0].index < secs[n+1]), pid] = True
                    else:
                        lns[1].loc[np.logical_and(lns[1].index > secs[n],
                                                  lns[1].index < secs[n+1]), pid] = True
    lineups = [{}, {}]
    line_df = pd.DataFrame(index=lns[0].index)
    for tt in range(2):
        n = 0
        for idx, row in lns[tt].iterrows():
            lu = [col for col in lns[tt].columns if row[col]]
            if len(lu) == 5:
                if len(lineups[tt]) == 0:
                    lineups[tt][n] = lu
                    n += 1
                elif not np.any([all_equal(lineups[tt][l], lu) for l in lineups[tt]]):
                    lineups[tt][n] = lu
                    n += 1
        lineupID = np.ones((lns[tt].shape[0],)) * -1
        for idx, row in lns[tt].iterrows():
            lu = [col for col in lns[tt].columns if row[col]]
            key = [l for l in lineups[tt] if all_equal(lineups[tt][l], lu)]
            lineupID[idx] = int(key[0]) if len(key) > 0 else -1
        if tt == 0:
            line_df['T_LineID'] = lineupID
        else:
            line_df['O_LineID'] = lineupID
    for idx, row in ev.iterrows():
        if row['EventTeamID'] == wid:
            if 'reb' in row['EventType']:
                line_df.loc[row['ElapsedSeconds'], 'T_R'] = 1
            if 'made' in row['EventType']:
                line_df.loc[row['ElapsedSeconds'], 'T_Pt'] = int(row['EventType'][-1])
                line_df.loc[row['ElapsedSeconds'], 'T_Shot'] = 1
                line_df.loc[row['ElapsedSeconds'], 'T_Poss'] = 1
            if 'miss' in row['EventType']:
                line_df.loc[row['ElapsedSeconds'], 'T_Shot'] = 1
                line_df.loc[row['ElapsedSeconds'], 'T_Poss'] = 1
            if 'turnover' in row['EventType']:
                line_df.loc[row['ElapsedSeconds'], 'T_TO'] = 1
                line_df.loc[row['ElapsedSeconds'], 'T_Poss'] = 1
        if row['EventTeamID'] == lid:
            if 'reb' in row['EventType']:
                line_df.loc[row['ElapsedSeconds'], 'O_R'] = 1
            if 'made' in row['EventType']:
                line_df.loc[row['ElapsedSeconds'], 'O_Pt'] = int(row['EventType'][-1])
                line_df.loc[row['ElapsedSeconds'], 'O_Shot'] = 1
                line_df.loc[row['ElapsedSeconds'], 'O_Poss'] = 1
            if 'miss' in row['EventType']:
                line_df.loc[row['ElapsedSeconds'], 'O_Shot'] = 1
                line_df.loc[row['ElapsedSeconds'], 'O_Poss'] = 1
            if 'turnover' in row['EventType']:
                line_df.loc[row['ElapsedSeconds'], 'O_TO'] = 1
                line_df.loc[row['ElapsedSeconds'], 'O_Poss'] = 1
    line_df['Mins'] = 1 / 60
    line_df = line_df.fillna(0)
    lsum_dfs = (line_df.groupby(['T_LineID']).sum().drop(columns=['O_LineID']),
                line_df.groupby(['O_LineID']).sum().drop(columns=['T_LineID']),
                line_df.groupby(['T_LineID', 'O_LineID']).sum())
    for col in line_df.columns.drop(['T_LineID', 'O_LineID', 'Mins']):
        for n in range(3):
            lsum_dfs[n][col + 'PM'] = lsum_dfs[n][col] - lsum_dfs[n]['O_' + col[2:]] if col[:2] == 'T_' else \
                lsum_dfs[n][col] - lsum_dfs[n]['T_' + col[2:]]
    
    return ev, lineups, line_df, lsum_dfs

def getGameLineups(files, season):
    evp = pd.read_csv(files['MEvents{}'.format(season)]).drop(columns=['Season']).set_index(['DayNum', 'WTeamID', 'LTeamID', 'EventPlayerID'])
    evp['GameID'] = evp.groupby(['DayNum', 'WTeamID', 'LTeamID']).ngroup()
    evp = evp.reset_index().set_index(['GameID', 'EventID'])
    evp = evp.loc[evp['WTeamID'] == 1140]
    evp = evp.drop(columns=['X', 'Y', 'Area'])
    lineup_ids = {}
    game_df = pd.DataFrame()
    
    for (gid, wid, lid), ev in tqdm(evp.groupby(['GameID', 'WTeamID', 'LTeamID'])):
        ev = ev.loc[ev['EventPlayerID'] != 0]
        mxepsec = ev['ElapsedSeconds'].max()
        ov = [0, 1200, 2400]
        while ov[-1] < mxepsec:
            ov = ov + [ov[-1] + 300]
        lns = (pd.DataFrame(index=np.arange(ov[-1]), columns=np.sort(list(set(ev.loc[ev['EventTeamID'] == wid,
                                                                             'EventPlayerID'].values))),
                             data=False),
               pd.DataFrame(index=np.arange(ov[-1]), columns=np.sort(list(set(ev.loc[ev['EventTeamID'] == lid,
                                                                             'EventPlayerID'].values))),
                             data=False))
        for pid, p in ev.groupby(['EventPlayerID']):
            if pid != 0:
                poss_subs = np.sort(ov + \
                    list(p.loc[p['EventSubType'] == 'in', 'ElapsedSeconds'] - .5) + \
                        list(p.loc[p['EventSubType'] == 'out', 'ElapsedSeconds'] + .5))
                nev, secs = np.histogram(p['ElapsedSeconds'], poss_subs)
                secs = secs.astype(int)
                for n in range(len(nev)):
                    if nev[n] > 0:
                        if pid in lns[0].columns:
                            lns[0].loc[np.logical_and(lns[0].index > secs[n],
                                                      lns[0].index < secs[n+1]), pid] = True
                        else:
                            lns[1].loc[np.logical_and(lns[1].index > secs[n],
                                                      lns[1].index < secs[n+1]), pid] = True
        line_df = pd.DataFrame(index=lns[0].index)
        for tt in range(2):
            if wid in lineup_ids and tt == 0:
                lineups = lineup_ids[wid]
            elif lid in lineup_ids and tt == 1:
                lineups = lineup_ids[lid]
            else:
                lineups = []
            for idx, row in lns[tt].iterrows():
                lu = [col for col in lns[tt].columns if row[col]]
                if len(lu) == 5:
                    if lu not in lineups:
                        lineups.append(lu)
                        lu_id = len(lineups) - 1
                    else:
                        lu_id = [i for i, d in enumerate(lineups) if lu == d][0]
                else:
                    lu_id = -1
                if tt == 0:
                    line_df.loc[idx, 'T_LineID'] = lu_id
                else:
                    line_df.loc[idx, 'O_LineID'] = lu_id
            if tt == 0:
                lineup_ids[wid] = lineups
            else:
                lineup_ids[lid] = lineups
        for idx, row in ev.iterrows():
            if row['EventTeamID'] == wid:
                if 'reb' in row['EventType']:
                    line_df.loc[row['ElapsedSeconds'], 'T_R'] = 1
                if 'made' in row['EventType']:
                    line_df.loc[row['ElapsedSeconds'], 'T_Pt'] = int(row['EventType'][-1])
                    line_df.loc[row['ElapsedSeconds'], 'T_Shot'] = 1
                    line_df.loc[row['ElapsedSeconds'], 'T_Poss'] = 1
                if 'miss' in row['EventType']:
                    line_df.loc[row['ElapsedSeconds'], 'T_Shot'] = 1
                    line_df.loc[row['ElapsedSeconds'], 'T_Poss'] = 1
                if 'turnover' in row['EventType']:
                    line_df.loc[row['ElapsedSeconds'], 'T_TO'] = 1
                    line_df.loc[row['ElapsedSeconds'], 'T_Poss'] = 1
            if row['EventTeamID'] == lid:
                if 'reb' in row['EventType']:
                    line_df.loc[row['ElapsedSeconds'], 'O_R'] = 1
                if 'made' in row['EventType']:
                    line_df.loc[row['ElapsedSeconds'], 'O_Pt'] = int(row['EventType'][-1])
                    line_df.loc[row['ElapsedSeconds'], 'O_Shot'] = 1
                    line_df.loc[row['ElapsedSeconds'], 'O_Poss'] = 1
                if 'miss' in row['EventType']:
                    line_df.loc[row['ElapsedSeconds'], 'O_Shot'] = 1
                    line_df.loc[row['ElapsedSeconds'], 'O_Poss'] = 1
                if 'turnover' in row['EventType']:
                    line_df.loc[row['ElapsedSeconds'], 'O_TO'] = 1
                    line_df.loc[row['ElapsedSeconds'], 'O_Poss'] = 1
        line_df['Mins'] = 1 / 60
        line_df = line_df.fillna(0)
        for lnn in ['T_LineID', 'O_LineID']:
            opp = 'O_LineID' if 'T_' in lnn else 'T_LineID'
            lsum_df = line_df.groupby(lnn).sum().drop(columns=opp)
            for col in lsum_df.columns.drop(['Mins']):
                lsum_df[col + 'PM'] = lsum_df[col] - lsum_df['O_' + col[2:]] if col[:2] == 'T_' else \
                    lsum_df[col] - lsum_df['T_' + col[2:]]
            lsum_df['GameID'] = gid
            lsum_df = lsum_df.reset_index()
            if 'O_' in lnn:
                lsum_df['TID'] = lid
                lsum_df.columns = \
                    ['T_' + col[2:] if 'O_' in col else 'O_' + col[2:] if 'T_' in col else col for col in lsum_df.columns]
            else:
                lsum_df['TID'] = wid
            game_df = game_df.append(lsum_df.set_index(['GameID', 'TID', 'T_LineID']))
    liddf = pd.DataFrame()
    for key in lineup_ids:
        tmp = pd.DataFrame(lineup_ids[key])
        tmp['LineID'] = np.arange(tmp.shape[0])
        tmp['TID'] = key
        liddf = liddf.append(tmp.set_index(['TID', 'LineID']))

def getAdvStats(df):
    wdf = pd.DataFrame(index=df.index)
    mins = df['Mins']
    mins[mins < 5] = 5 #Removes exploding values
    wdf['AstPer18'] = df['Ast'] * 18 / mins.values
    wdf['PtsPer18'] = df['Pts'] * 18 / mins.values
    wdf['BlkPer18'] = df['Blk'] * 18 / mins.values
    wdf['StlPer18'] = df['Stl'] * 18 / mins.values
    wdf['TOPer18'] = df['TO'] * 18 / mins.values
    wdf['FoulPer18'] = df['Foul'] * 18 / mins.values
    wdf['ORPer18'] = df['OR'] * 18 / mins.values
    wdf['DRPer18'] = df['DR'] * 18 / mins.values
    wdf['RPer18'] = (df['OR'] + df['DR']) * 18 / mins.values
    wdf['Lay%'] = df['LayM'] / df['FGM']
    wdf['Dunk%'] = df['DunkM'] / df['FGM']
    wdf['FG%'] = df['FGM'] / df['FGA']
    wdf['3Pt%'] = df['FGM3'] / df['FGA3']
    wdf['eFG%'] = (df['FGM'] + .5 * df['FGM3']) / df['FGA']
    wdf['TS%'] = df['Pts'] / (2 * (df['FGA'] + .44 * df['FTA']))
    wdf['Econ'] = df['Ast'] + df['Stl'] - df['TO']
    wdf['EffShot%'] = (df['FGA3'] + df['LayA'] + df['DunkA']) / df['FGA']
    wdf['FT/A'] = df['FTA'] / df['FGA']
    wdf['FT%'] = df['FTM'] / df['FTA']
    return wdf.dropna()

def getRateStats(df, sdf, pdf):
    mdf = df.merge(pdf['TeamID'], on='PlayerID', right_index=True).sort_index()
    mdf = mdf.merge(sdf, left_on=['GameID', 'TeamID'], right_on=['GameID', 'TID'], right_index=True)
    wdf = pd.DataFrame(index=mdf.index)
    scaling = 67 / (mdf['Mins'] / \
            (poss_breaks[np.digitize(mdf['Mins'] * 60, poss_breaks[2:]) + 2] / 60) * mdf['T_Poss'])
    wdf['AstScale'] = mdf['Ast'] * scaling
    wdf['PtsScale'] = mdf['Pts'] * scaling
    wdf['RScale'] = (mdf['OR'] + mdf['DR']) * scaling
    wdf['EloDiff'] = mdf['T_Elo'] - mdf['O_Elo']
    wdf['R%'] = (mdf['OR'] + mdf['DR']) / (mdf['T_OR'] + mdf['T_DR'])
    wdf['Pts%'] = mdf['Pts'] / mdf['T_Score']
    wdf['EloScaling'] = 1 / abs(wdf['EloDiff'] * scaling / mdf['T_Poss'])
    return wdf.dropna()

def getPlayerSeasonStats(df):
    return df.groupby(['PlayerID']).mean()

def getTeamRosters(files):
    df = pd.read_csv(files['MPlayers'], error_bad_lines=False)
    df['FullName'] = df['FirstName'] + ' ' + df['LastName']
    df = df.set_index(['PlayerID'])
    return df

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
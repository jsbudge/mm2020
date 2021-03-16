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
import statslib as st

def all_equal(i, q):
    return np.all([ii in q for ii in i])

poss_breaks = np.array([0, 1200, 2400, 2700, 3000, 3300, 3600, 3900, 4200])

'''
getRosters
Gets stats for every player in every game.
Inputs:
    season - (int) End Year of season (i.e., for 2017-18 season, this would be 2018)
Returns:
    teams - (DataFrame) Frame of game ids with team ids and daynums for
                        correct association of games with stats.
    evst - (DataFrame) Frame of all the stats for each player, arranged by season
                        and team.
'''
def getRosters(season):
    evp = pd.read_csv('./data/MEvents{}.csv'.format(season)).drop(columns=['Season']).set_index(['DayNum', 'WTeamID', 'LTeamID', 'EventPlayerID']).fillna(0)
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

'''
getSingleGameLineups
Gets stats for lineups in a single game.
Inputs:
    gid - (int) GameID of single game to look at.
    season - (int) End Year of season (i.e., for 2017-18 season, this would be 2018)
Returns:
    ev - (DataFrame) Frame of all events recorded during the game.
    lineups - (list) List with two dicts inside, containing the PlayerIDs for
                        all players in the indicated lineup. LineupIDs are used
                        as the keys to each dict.
    line_df - (DataFrame) Frame of second-by-second events taken by each lineup.
    lsum_dfs - (tuple) Tuple of DataFrames, giving the stats of each lineup. Each
                        tuple entry is the frame for one team's lineups, the last
                        being the intersection of both lineups.
'''
def getSingleGameLineups(gid, season):
    evp = pd.read_csv('./data/MEvents{}.csv'.format(season)).drop(columns=['Season']).set_index(['DayNum', 'WTeamID', 'LTeamID', 'EventPlayerID'])
    evp['GameID'] = evp.groupby(['DayNum', 'WTeamID', 'LTeamID']).ngroup()
    evp = evp.reset_index().set_index(['GameID', 'EventID'])
    ev = evp.loc(axis=0)[gid, :].copy()
    wid = ev['WTeamID'].values[0]; lid = ev['LTeamID'].values[0]
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

def getGameLineups(season):
    evp = pd.read_csv('./data/MEvents{}.csv'.format(season)).drop(columns=['Season']).set_index(['DayNum', 'WTeamID', 'LTeamID', 'EventPlayerID'])
    evp['GameID'] = evp.groupby(['DayNum', 'WTeamID', 'LTeamID']).ngroup()
    evp = evp.reset_index().set_index(['GameID', 'EventID'])
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
        
def getLineupsWithoutStats(season):
    sdata = st.getGames(season, split=True)[0]
    evp = pd.read_csv('./data/MEvents{}.csv'.format(season)).drop(columns=['Season'])
    evp = evp.merge(sdata[['GameID', 'DayNum', 'TID', 'OID', 'NumOT']], left_on=['DayNum', 'WTeamID', 'LTeamID'], right_on=['DayNum', 'TID', 'OID'])
    evp = evp.drop(columns=['X', 'Y', 'Area', 'TID', 'OID', 'DayNum']).reset_index().set_index(['GameID', 'EventID'])
    evp = evp.drop(columns=['index'])
    evp = evp.loc[evp['EventPlayerID'] != 0].fillna('')
    evp.loc[evp['EventSubType'] == 'unk', 'EventSubType'] = ''
    evp['EventType'] = evp['EventType'] + evp['EventSubType']
    evp = evp.drop(columns=['EventSubType'])
    sdata = sdata.set_index(['GameID'])
    subins = evp.loc[evp['EventType'] == 'subin'].groupby(['GameID', 'EventPlayerID']).apply(lambda x: list(x['ElapsedSeconds'].values - .5))
    subouts = evp.loc[evp['EventType'] == 'subout'].groupby(['GameID', 'EventPlayerID']).apply(lambda x: list(x['ElapsedSeconds'].values + .5))
    otherev = evp.groupby(['GameID', 'EventPlayerID']).apply(lambda x: list(x['ElapsedSeconds'].values))
    gamestops = evp.groupby(['GameID']).apply(lambda x: [0, 1200] + [2400 + 300 * n for n in range(x['NumOT'].values[0] + 1)])
    subdata = pd.DataFrame(index=subins.index.append(subouts.index.append(otherev.index))).reset_index().drop_duplicates()
    subdata = subdata.set_index(['GameID', 'EventPlayerID'])
    subdata.loc[subins.index, 'in'] = subins
    subdata.loc[subouts.index, 'out'] = subouts
    subdata.loc[otherev.index, 'ev'] = otherev
    for col in subdata.columns:
        subdata.loc[subdata.isnull()[col], col] = subdata.loc[subdata.isnull()[col], col].apply(lambda x: [])
    lineup_ids = {}
    game_df = pd.DataFrame()
    for tid in tqdm(list(set(evp[['WTeamID', 'LTeamID']].values.flatten()))):
        lines = []
        act_df = evp.loc[np.logical_or(evp['WTeamID'] == tid,
                                       evp['LTeamID'] == tid)]
        act_df = act_df.loc[act_df['EventTeamID'] == tid]
        players = np.sort(list(set(act_df['EventPlayerID'].values)))
        for gid, ev in act_df.groupby(['GameID']):
            ov = gamestops.loc[gid]
            gr = np.arange(ov[-1])
            lns = pd.DataFrame(index=gr, columns=players,
                                 data=False)
            for idx, row in subdata.loc[gid].iterrows():
                if idx in lns.columns:
                    poss_subs = np.sort(ov + row['in'] + row['out'])
                    nev, secs = np.histogram(row['ev'], poss_subs)
                    for n in range(len(nev)):
                        if nev[n] > 0:
                            lns.loc[np.logical_and(gr > secs[n],
                                                   gr < secs[n+1]), idx] = True
            
            ll = np.zeros((ov[-1],))
            for idx, row in lns.iterrows():
                lu = [col for col in lns.columns if row[col]]
                if len(lu) == 5:
                    if lu not in lines:
                        lines.append(lu)
                        lu_id = len(lines) - 1
                    else:
                        lu_id = [i for i, d in enumerate(lines) if lu == d][0]
                else:
                    lu_id = -1
                ll[idx] = lu_id
            line_df = pd.DataFrame(index=gr)
            line_df['T_LineID'] = ll
            line_df['Mins'] = 1 / 60
            line_df['GameID'] = gid
            line_df['TID'] = tid
            line_df = line_df.loc[line_df['T_LineID'] != -1]
            game_df = game_df.append(line_df.groupby(['GameID', 'TID', 'T_LineID']).sum())
        lineup_ids[tid] = lines
    return lineup_ids, game_df

'''
getAdvStats
Given a raw stat frame, gives some more advanced statistics.
Inputs:
    df - (DataFrame) Frame of basic stats, taken from getRosters above.
Returns:
    wdf - (DataFrame) Frame with advanced statistics, using the same indexes as
                        df.
'''
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

'''
splitStats
Arranges internet data to be of most use.
Inputs:
    df - (DataFrame) Frame from internet data, generally taken from InternetPlayerData.csv.
    sdf - (DataFrame) Frame of seasonal stats, from statslib's getSeasonalStats.
    add_stats - (list) List of additional options:
                            mins: Get stats adjusted for minutes played.
                            poss: Get stats adjusted for possessions played.
                        An empty list will not have any of these additional stats.
    minbins - (int) or (list) If an int, the number of bins to use in MinPerc,
                    divided evenly. If a list, specifies the bins used for MinPerc binning.
Returns:
    adv_df - (DataFrame) Frame of advanced stats.
    phys_df - (DataFrame) Frame of housekeeping stats, such as height, weight, and
                            minutes played.
    score_df - (DataFrame) Frame of overall scoring stats. May not return anything.
    base_df - (DataFrame) Frame of basic stats, or everything not in the other
                            frames.
    Each frame will have indexes equal to df.
'''
def splitStats(df, sdf, add_stats=[], minbins=None):
    df.loc[df['GP'] == 0, 'GP'] = 1
    df.loc[df['FGA'] == 0, 'FGA'] = 1
    df['MinsPerGame'] = df['Mins'] / df['GP']
    if minbins is not None:
        if type(minbins) == int:
            cats = np.percentile(df['MinsPerGame'], np.linspace(0, 100, minbins+1))
            cats[-1] = cats[-1] + 1
        else:
            cats = minbins
    df['MinPerc'] = np.digitize(df['MinsPerGame'], cats)
    mdf = df.join(sdf[['T_SoS', 'T_Poss']], on=['Season', 'TID'])
    adv_df = df[['Ast%', 'Blk%', 'BPM', 'DBPM', 'DR%', 'DWS', 'eFG%', 'FG%', 
                 'FT/A', 'FT%', 'OBPM', 'OR%', 'OWS', 'PER', 'PtsProd',
                 'Stl%', '3/2Rate', 'FG3%', 'R%', 'TS%', 'TO%', 'FG2%', 'Usage%',
                 'WS']].copy()
    adv_df['Econ'] = (df['Ast'] + df['Stl'] - df['TO']).values
    adv_df['PPS'] = ((df['Pts'] - df['FTM']) / df['FGA']).values
    adv_df['GameSc'] = (40 * df['eFG%'] + 2 * df['R%'] + 15 * df['FT/A'] + 25 - 2.5 * df['TO%']).values
    adv_df['SoS'] = mdf['T_SoS'].values
    phys_df = df[['Pos', 'Height', 'Weight', 'GP', 'GS', 'Mins', 'MinPerc', 'MinsPerGame']].copy()
    phys_df['T_Poss'] = mdf['T_Poss'].values
    score_df = df[[col for col in df if 'Score' in col]].copy()
    base_df = df[[col for col in df if col not in adv_df.columns and col not in phys_df.columns and col not in score_df.columns]].drop(columns=['WSPer40'])
    
    #The 75th percentile of players spend 26 minutes on court, so let's adjust to that
    if len(add_stats) > 0:
        adj_mins = phys_df['Mins'].values
        adj_mins[adj_mins < 5] = 5 #cutoff to avoid exploding numbers
        adj_poss = mdf['T_Poss'] / 40 * phys_df['MinsPerGame']
        adj_poss.loc[phys_df['MinsPerGame'] < 5] = 1
        for col in base_df:
            if 'FG' not in col and 'FT' not in col:
                if 'mins' in add_stats:
                    adv_df[col + 'Per26Min'] = base_df[col] * 26 / adj_mins
                if 'poss' in add_stats:
                    adv_df[col + 'Per100Poss'] = base_df[col] / (phys_df['GP'] * adj_poss) * 100
            
    return adv_df, phys_df, score_df, base_df

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

'''
getTeamRosters
Gets stats for every player in every game.
Returns:
    df - (DataFrame) Frame of player names, teams and IDs. Index is PlayerID.
'''
def getTeamRosters():
    df = pd.read_csv('./data/MPlayers.csv', error_bad_lines=False)
    df['FullName'] = df['FirstName'] + ' ' + df['LastName']
    df = df.set_index(['PlayerID'])
    return df

'''
loadPlayerNames
Gets handy dict for finding player names from IDs and vice versa.
Returns:
    ret - (dict) Dict where keys include player IDs and names.
'''

def loadPlayerNames():
    df = pd.read_csv('./data/MPlayers.csv')
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
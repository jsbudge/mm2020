'''

This is the statistics library that we'll be using.

'''

import numpy as np
import pandas as pd
import os

stat_names = ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 
              'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
id_cols = ['GameID', 'Season', 'DayNum', '1_TeamID', '2_TeamID']

def getFiles(fdir='data'):
    data = {}
    for file in os.listdir(fdir):
        if file.endswith(".csv"):
            dict_name = file.split('.')[0]
            data[dict_name] = os.path.join(fdir, file)
    return data

def getGames(fnme, both_teams=True, duplicate=False):
    df = pd.read_csv(fnme)
    df['WLoc'] = df['WLoc'].map({'A': -1, 'N': 0, 'H': 1})
    df['GameID'] = df.index.values
    wdf = df[['GameID', 'Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc',
       'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
       'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].rename(columns={'WLoc': 'GLoc'})
    ldf = df[['GameID', 'Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc',
    'NumOT', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst',
    'LTO', 'LStl', 'LBlk', 'LPF']].rename(columns={'WLoc': 'GLoc'})
    ldf['GLoc'] = -ldf['GLoc']
    
    if not both_teams:
        for col in wdf.columns:
            if col[0] == 'W':
                wdf = wdf.rename(columns={col: col[1:]})
        for col in ldf.columns:
            if col[0] == 'L':
                ldf = ldf.rename(columns={col: col[1:]})
        if duplicate:
            return wdf.append(ldf, ignore_index=True).dropna(axis=1)
        else:
            return wdf, ldf
    else:
        wdf_1 = wdf.copy(); wdf_2 = wdf.copy()
        ldf_1 = ldf.copy(); ldf_2 = ldf.copy()
        for col in df.columns:
            if col[0] == 'W':
                wdf_1 = wdf_1.rename(columns={col: '1_' + col[1:]})
                wdf_2 = wdf_2.rename(columns={col: '2_' + col[1:]})
                ldf_1 = ldf_1.rename(columns={col: '1_' + col[1:]})
                ldf_2 = ldf_2.rename(columns={col: '2_' + col[1:]})
            if col[0] == 'L':
                ldf_1 = ldf_1.rename(columns={col: '2_' + col[1:]})
                ldf_2 = ldf_2.rename(columns={col: '1_' + col[1:]})
                wdf_1 = wdf_1.rename(columns={col: '2_' + col[1:]})
                wdf_2 = wdf_2.rename(columns={col: '1_' + col[1:]})
        if duplicate:
            ret = wdf_1.merge(ldf_1, on='GameID').append(wdf_2.merge(ldf_2, on='GameID'), ignore_index=True)
            for col in ret.columns:
                if col[:-2] == '_x':
                    ret = ret.rename(columns={col: col[:-2]})
                elif col[:-2] == '_y':
                    ret = ret.drop(columns=[col])
            return ret
        else:
            w = wdf_1.merge(ldf_1, on=id_cols)
            l = wdf_2.merge(ldf_2, on=id_cols)
            for col in w.columns:
                if col[-2:] == '_x':
                    w = w.rename(columns={col: col[:-2]})
                    l = l.rename(columns={col: col[:-2]})
                elif col[-2:] == '_y':
                    w = w.drop(columns=[col])
                    l = l.drop(columns=[col])
            return w, l
         
def normalizeToSeason(df):
    for season, sdf in df.groupby(['Season']):
        for col in sdf.columns:
            if col not in ['GameID', 'Season', 'GLoc', 'DayNum', 'TeamID', '1_TeamID', '2_TeamID', 'NumOT']:
                df.loc[df['Season'] == season, col] = (sdf[col].values - sdf[col].mean()) / sdf[col].std()
    return df

def getDiffs(df):
    ret = pd.DataFrame()
    for st in stat_names:
        ret[st + '_diff'] = df['1_' + st] - df['2_' + st]
    for ids in id_cols:
        ret[ids] = df[ids]
    return ret

def getRatios(df):
    ret = pd.DataFrame()
    for st in stat_names:
        ret[st + '_perc'] = df['1_' + st] / df['2_' + st]
    for ids in id_cols:
        ret[ids] = df[ids]
    return ret

def addStats(df):
    df['FG%'] = df['FGM'] / df['FGA']
    df['PPS'] = (df['Score'] - df['FTM']) / df['FGA']
    df['eFG%'] = (df['FGM'] + .5 * df['FGM3']) / df['FGA']
    df['TS%'] = df['Score'] / (2 * (df['FGA'] + .44 * df['FTA']))
    df['Econ'] = df['Ast'] + df['Stl'] - df['TO']
    df['Poss'] = .96 * (df['FGA'] - df['OR'] + df['TO'] + .44 * df['FTA'])
    df['OffRat'] = df['Score'] * 100 / df['Poss']
    return df

def getSeasonMeans(df):
    return df.groupby(['Season', 'TeamID']).mean().drop(columns=['GameID', 'GLoc', 'DayNum'])

def getSeasonVars(df):
    return df.groupby(['Season', 'TeamID']).std().drop(columns=['GameID', 'GLoc', 'DayNum'])

def loadTeamNames(file_dict):
    df = pd.read_csv(file_dict['MTeams'])
    ret = {}
    for idx, row in df.iterrows():
        ret[row['TeamID']] = row['TeamName']
        ret[row['TeamName']] = row['TeamID']
    return ret

def addMasseyOrdinals(df, files):
    mo = pd.read_csv(files['MMasseyOrdinals'])
    
        
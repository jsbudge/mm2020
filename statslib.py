'''

This is the statistics library that we'll be using.

'''

import numpy as np
import pandas as pd
import os

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
    wdf = df[['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc',
       'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
       'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].rename(columns={'WLoc': 'GLoc'})
    ldf = df[['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc',
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
            return wdf.append(ldf).dropna(axis=1)
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
            return wdf_1.merge(ldf_1).append(wdf_2.merge(ldf_2))
        else:
            return wdf_1.merge(ldf_1), wdf_2.merge(ldf_2)
         
def normalizeToSeason(df):
    for season, sdf in df.groupby(['Season']):
        for col in sdf.columns:
            if col not in ['Season', 'GLoc', 'DayNum', 'TeamID', '1_TeamID', '2_TeamID', 'NumOT']:
                df.loc[df['Season'] == season, col] = sdf[col].values / sdf[col].mean() - 1
    return df
        
'''

This is the statistics library that we'll be using.

'''

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from scipy import linalg
from math import ceil

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
    
def getSystemWeights(files):
    #This should only need to be run once, since it saves its results out to a CSV
    wdf = getGames(files['MRegularSeasonDetailedResults'], True, False)[0]
    wdf_full = getGames(files['MRegularSeasonDetailedResults'], False, True)
    mo = pd.read_csv(files['MMasseyOrdinals'])
    weights_dict= {}
    for season in np.arange(2011, 2020):
        wdf2019 = wdf_full.loc[wdf_full['Season'] == season]
        mo2019 = mo.loc[mo['Season'] == season]
        
        ranksys = {}; weights_dict[season] = {}
        
        '''
        WEIGHTING SYSTEM FOR MASSEY ORDINALS
        '''
        for idx, sys in tqdm(mo2019.groupby(['SystemName'])):
            ranksys[idx] = {}
            ranksys[idx]['rankscore'] = 0
            for tid, team in sys.groupby(['TeamID']):
                if team.shape[0] < 2:
                    ranksys[idx][tid] = team['OrdinalRank'].values[0] * np.ones((wdf2019.loc[wdf2019['TeamID'] == tid].shape[0],))
                else:
                    fuunc = CubicSpline(team['RankingDayNum'], team['OrdinalRank'], bc_type='clamped')(wdf2019.loc[wdf2019['TeamID'] == tid, 'DayNum'])
                    fuunc[wdf2019.loc[wdf2019['TeamID'] == tid, 'DayNum'] < team['RankingDayNum'].values[0]] = team['OrdinalRank'].values[0]
                    ranksys[idx][tid] = fuunc
                                                                                                                               
                    
        
        wdf_diffs = getDiffs(wdf.loc[wdf['Season'] == season])
        max_score = wdf_diffs['Score_diff'].max()
        for idx, row in tqdm(wdf_diffs.iterrows()):
            for sys in ranksys:
                try:
                    ranksys[sys]['rankscore'] -= \
                        (ranksys[sys][row['1_TeamID']][wdf2019.loc[wdf2019['TeamID'] == row['1_TeamID'], 'DayNum'] == row['DayNum']][0] \
                         - ranksys[sys][row['2_TeamID']][wdf2019.loc[wdf2019['TeamID'] == row['2_TeamID'], 'DayNum'] == row['DayNum']][0]) / wdf_diffs.shape[0] * row['Score_diff'] / max_score
                except:
                    ranksys[sys]['rankscore'] -= 0
                    
        
        for key in ranksys:
            weights_dict[season][key] = ranksys[key]['rankscore']
    
    sys_weights = pd.DataFrame(index=list(set(mo['SystemName'])), columns=np.arange(2011, 2020))
    for sys in list(set(mo['SystemName'])):
        for season in np.arange(2011, 2020):
            if sys in weights_dict[season]:
                sys_weights.loc[sys, season] = weights_dict[season][sys]
    sys_weights += abs(np.nanmin(sys_weights.astype(np.float64).values)) + 1
    #sys_weights['SystemName'] = sys_weights.index
    
    sys_weights.to_csv('sys_weights.csv', index_label='SystemName')
    return sys_weights

def getRanks(files):
    wdf = getGames(files['MRegularSeasonDetailedResults'], False, True)
    wdf = wdf.loc[wdf['Season'] >= 2011].sort_values('DayNum')
    weights = pd.read_csv('sys_weights.csv')
    wdf['Rank'] = 999
    mo = pd.read_csv(files['MMasseyOrdinals'])
    
    for idx, grp in tqdm(wdf.groupby(['Season', 'TeamID'])):
        grp = grp.sort_values('GameID')
        ranks = mo.loc[np.logical_and(mo['Season'] == idx[0], mo['TeamID'] == idx[1])].merge(weights, on='SystemName').sort_values('RankingDayNum')
        wdf.loc[np.logical_and(wdf['Season'] == idx[0], 
                               wdf['TeamID'] == idx[1]), 'Rank'] = lowess(ranks['RankingDayNum'].values, 
                                       ranks['OrdinalRank'].values, 
                                       ranks[str(idx[0])].values, x0=grp['DayNum'], f=.15)
    return wdf

def lowess(x, y, w, x0=None, f=.1, n_iter=3):
    """lowess_bell_shape_kern(x, y, tau = .005) -> yest
    Locally weighted regression: fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y at x0 (optional).
    """
    
    n = len(x)
    r = int(np.ceil(f * n))
    yest = np.zeros(n)
    delta = np.ones(n)
    
    #Looping through all x-points
    for iteration in range(n_iter):
        for i in range(n):
            weights = np.zeros((n,))
            weights[max(0, i-r):min(i+r, n)] = w[max(0, i-r):min(i+r, n)]
            weights *= delta
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                        [np.sum(weights * x), np.sum(weights * x * x)]])
            theta = linalg.solve(A, b)
            yest[i] = theta[0] + theta[1] * x[i] 
        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2
    if x0 is None:
        return yest
    else:
        return np.interp(x0, x, yest)
    
        
'''

This is the statistics library that we'll be using.

'''

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from scipy import linalg

stat_names = ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 
              'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
id_cols = ['GameID', 'Season', 'DayNum', 'T_TeamID', 'O_TeamID']


'''
getFiles
Creates a dict of names for easy file access. Does this in an operating-system
agnostic way so we can work in Python and Windows.

Params:
    fdir: String - Directory to search for files.

Returns:
    data: Dict - index: file name, each element is the path to that file
'''
def getFiles(fdir='data'):
    data = {}
    for file in os.listdir(fdir):
        if file.endswith(".csv"):
            dict_name = file.split('.')[0]
            data[dict_name] = os.path.join(fdir, file)
    return data

'''
getGames
Arranges a DataFrame to be more friendly to statistical analysis.

Params:
    fnme: String - path to CSV file
    split: bool - Determines whether to return a single, double size DataFrame or
                    a tuple of DataFrames arranged by winners and losers
                
Returns:
    split = True
        w: DataFrame - frame arranged as T_ = winners of games
        l: DataFrame - frame arranged as T_ = losers of games
    split = False
        ret: DataFrame - frame of all games, double the length of the original CSV
'''
def getGames(fnme, split=False):
    df = pd.read_csv(fnme)
    
    #Map location to numbers
    df['WLoc'] = df['WLoc'].map({'A': -1, 'N': 0, 'H': 1})
    
    #Add a GameID so we can identify all the games with a single ID
    df['GameID'] = df.index.values
    
    #Grab the correct columns for winners and losers
    wdf = df[['GameID', 'Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc',
       'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
       'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']].rename(columns={'WLoc': 'GLoc'})
    ldf = df[['GameID', 'Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc',
    'NumOT', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst',
    'LTO', 'LStl', 'LBlk', 'LPF']].rename(columns={'WLoc': 'GLoc'})
    
    #Flip location for losing teams, since 'Home' and 'Away' are based on the winning team
    ldf['GLoc'] = -ldf['GLoc']
    
    #We need to copy things like this so each team's winning and losing games
    #are preserved
    wdf_1 = wdf.copy(); wdf_2 = wdf.copy()
    ldf_1 = ldf.copy(); ldf_2 = ldf.copy()
    
    #Remove winner and loser columns and replace with T or O
    for col in df.columns:
        if col[0] == 'W':
            wdf_1 = wdf_1.rename(columns={col: 'T_' + col[1:]})
            wdf_2 = wdf_2.rename(columns={col: 'O_' + col[1:]})
            ldf_1 = ldf_1.rename(columns={col: 'T_' + col[1:]})
            ldf_2 = ldf_2.rename(columns={col: 'O_' + col[1:]})
        if col[0] == 'L':
            ldf_1 = ldf_1.rename(columns={col: 'O_' + col[1:]})
            ldf_2 = ldf_2.rename(columns={col: 'T_' + col[1:]})
            wdf_1 = wdf_1.rename(columns={col: 'O_' + col[1:]})
            wdf_2 = wdf_2.rename(columns={col: 'T_' + col[1:]})
    if not split:
        ret = wdf_1.merge(ldf_1, on='GameID').append(wdf_2.merge(ldf_2, on='GameID'), ignore_index=True)
        
        #Remove the extra stuff from columns that both teams share
        #Pandas adds extra identifiers during the dataframe merge
        for col in ret.columns:
            if col[-2:] == '_x':
                ret = ret.rename(columns={col: col[:-2]})
            elif col[-2:] == '_y':
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
         
'''
normalizeToSeason
Modifies a DataFrame so each statistical columns has zero mean and a standard
deviation of one. Modifies in-place.

Params:
    df: DataFrame - frame to normalize.
    
Returns:
    df: DataFrame - normalized frame.

This modifies the DataFrame in place and returns it.
'''
def normalizeToSeason(df):
    for season, sdf in df.groupby(['Season']):
        for col in sdf.columns:
            if col not in ['GameID', 'Season', 'GLoc', 'DayNum', 'T_TeamID', 'O_TeamID', 'NumOT']:
                df.loc[df['Season'] == season, col] = (sdf[col].values - sdf[col].mean()) / sdf[col].std()
    return df

'''
getDiffs
Get differences of each statistical column. Modifies in-place.

Params:
    df: DataFrame - getGames frame.

Returns:
    ret: DataFrame - frame with differences. Names are changed to reflect the
                        differencing.
'''
def getDiffs(df):
    ret = pd.DataFrame()
    for st in stat_names:
        ret[st + '_diff'] = df['T_' + st] - df['O_' + st]
    for ids in id_cols:
        ret[ids] = df[ids]
    return ret

'''
addStats
Adds various advanced statistical columns to a DataFrame. Modifies in-place.

Params:
    df: DataFrame - getGames frame.
    
Returns:
    df: DataFrame - augmented frame.
'''
def addStats(df):
    df['T_FG%'] = df['T_FGM'] / df['T_FGA']
    df['T_PPS'] = (df['T_Score'] - df['T_FTM']) / df['T_FGA']
    df['T_eFG%'] = (df['T_FGM'] + .5 * df['T_FGM3']) / df['T_FGA']
    df['T_TS%'] = df['T_Score'] / (2 * (df['T_FGA'] + .44 * df['T_FTA']))
    df['T_Econ'] = df['T_Ast'] + df['T_Stl'] - df['T_TO']
    df['T_Poss'] = .96 * (df['T_FGA'] - df['T_OR'] + df['T_TO'] + .44 * df['T_FTA'])
    df['T_OffRat'] = df['T_Score'] * 100 / df['T_Poss']
    df['T_R%'] = (df['T_OR'] + df['T_DR']) / (df['T_OR'] + df['T_DR'] + df['O_OR'] + df['O_DR'])
    df['T_Ast%'] = df['T_Ast'] / df['T_FGM']
    df['T_3Two%'] = df['T_FGA3'] / df['T_FGA']
    df['T_FT/A'] = df['T_FTA'] / df['T_FGA']
    df['T_FT%'] = df['T_FTM'] / df['T_FTA']
     
    df['O_FG%'] = df['O_FGM'] / df['O_FGA']
    df['O_PPS'] = (df['O_Score'] - df['O_FTM']) / df['O_FGA']
    df['O_eFG%'] = (df['O_FGM'] + .5 * df['O_FGM3']) / df['O_FGA']
    df['O_TS%'] = df['O_Score'] / (2 * (df['O_FGA'] + .44 * df['O_FTA']))
    df['O_Econ'] = df['O_Ast'] + df['O_Stl'] - df['O_TO']
    df['O_Poss'] = .96 * (df['O_FGA'] - df['O_OR'] + df['O_TO'] + .44 * df['O_FTA'])
    df['O_OffRat'] = df['O_Score'] * 100 / df['O_Poss']
    df['O_R%'] = 1 - df['T_R%']
    df['O_Ast%'] = df['O_Ast'] / df['O_FGM']
    df['O_3Two%'] = df['O_FGA3'] / df['O_FGA']
    df['O_FT/A'] = df['O_FTA'] / df['O_FGA']
    df['O_FT%'] = df['O_FTM'] / df['O_FTA']
    
    df['T_DefRat'] = df['O_OffRat']
    df['O_DefRat'] = df['T_OffRat']
    
    return df.fillna(0)

'''
getSeasonalStats
Calculates a weighted mean on each statistical column
and adds a few more stats that are valid for seasonal data.

Params:
    df: DataFrame - getGames frame, with columns added based on the weighting for the mean.
    
Returns:
    wdf: DataFrame - frame with a single entry per team per season, with the means
                        and added stats for that team.
'''
def getSeasonalStats(df):
    tcols = df.columns.drop(['Season', 'T_TeamID', 'GameID', 'GLoc', 'DayNum', 'O_TeamID'])
    wdf = df.groupby(['Season', 'T_TeamID']).mean().drop(columns=['GameID', 'GLoc', 'DayNum', 'O_TeamID'])
    wdf['T_Win%'] = 0
    wdf['T_PythWin%'] = 0
    wdf['T_SoS'] = 0
    for idx, grp in tqdm(df.groupby(['Season', 'T_TeamID'])):
        for col in tcols:
            wdf.loc[idx, col] = np.average(grp[col], weights=400 - grp['O_Rank'].values)
        wdf.loc[idx, ['T_Win%', 'T_PythWin%', 'T_SoS']] = [sum(grp['T_Score'] > grp['O_Score']) / grp.shape[0],
                                                           sum(grp['T_Score']**13.91) / sum(grp['T_Score']**13.91 + grp['O_Score']**13.91),
                                                           grp['O_Rank'].mean()]
    return wdf
        
'''
loadTeamNames
Look up a team and/or ID quickly and easily.

Params:
    file_dict: Dict - getFiles dict.
    
Returns:
    ret: Dict - dict: index is TeamID or TeamName, and elements have the TeamName
                        and TeamID, respectively.
'''
def loadTeamNames(file_dict):
    df = pd.read_csv(file_dict['MTeams'])
    ret = {}
    for idx, row in df.iterrows():
        ret[row['TeamID']] = row['TeamName']
        ret[row['TeamName']] = row['TeamID']
    return ret
    
'''
getSystemWeights
Given a getGames DataFrame and a file dict, returns a DataFrame of weights
where the index is the SystemName and the columns are the weighting of each
system during each season. A higher number = a better weighting.

Params:
    df: DataFrame - getGames frame.
    files: Dict - getFiles dict.

Returns:
    sys_weights: DataFrame - frame of weightings for ranking systems. Index is
                                SystemName and the columns are the weightings
                                corresponding to each season.
'''
def getSystemWeights(df, files):
    #This should only need to be run once, since it saves its results out to a CSV
    mo = pd.read_csv(files['MMasseyOrdinals'])
    weights_dict= {}
    for season in np.arange(2011, 2020):
        wdf_season = df.loc[df['Season'] == season]
        mo_season = mo.loc[mo['Season'] == season]
        
        ranksys = {}; weights_dict[season] = {}
        
        '''
        WEIGHTING SYSTEM FOR MASSEY ORDINALS
        '''
        for idx, sys in tqdm(mo_season.groupby(['SystemName'])):
            ranksys[idx] = {}
            
            #Initialize system's score
            ranksys[idx]['rankscore'] = 0
            for tid, team in sys.groupby(['T_TeamID']):
                if team.shape[0] < 2:
                    ranksys[idx][tid] = team['OrdinalRank'].values[0] * np.ones((wdf_season.loc[wdf_season['T_TeamID'] == tid].shape[0],))
                else:
                    #Interpolate between system outputs to get a team's rank at the time of an actual game
                    #This can be improved on, probably. Only using a Cubic Spline.
                    fuunc = CubicSpline(team['RankingDayNum'], team['OrdinalRank'], bc_type='clamped')(wdf_season.loc[wdf_season['T_TeamID'] == tid, 'DayNum'])
                    fuunc[wdf_season.loc[wdf_season['T_TeamID'] == tid, 'DayNum'] < team['RankingDayNum'].values[0]] = team['OrdinalRank'].values[0]
                    ranksys[idx][tid] = fuunc
                                                                                                                               
        wdf_diffs = getDiffs(wdf_season)
        max_score = wdf_diffs['Score_diff'].max()
        for idx, row in tqdm(wdf_diffs.iterrows()):
            for sys in ranksys:
                try:
                    #Modify rankscore by whether the system was correct AND
                    #how many points they were correct by (e.g., if the game
                    #was close or not
                    ranksys[sys]['rankscore'] -= \
                        (ranksys[sys][row['T_TeamID']][wdf_season.loc[wdf_season['T_TeamID'] == row['T_TeamID'], 'DayNum'] == row['DayNum']][0] \
                         - ranksys[sys][row['O_TeamID']][wdf_season.loc[wdf_season['O_TeamID'] == row['O_TeamID'], 'DayNum'] == row['DayNum']][0]) / wdf_diffs.shape[0] * row['Score_diff'] / max_score
                except:
                    ranksys[sys]['rankscore'] -= 0
                    
        
        for key in ranksys:
            weights_dict[season][key] = ranksys[key]['rankscore']
    
    #Arrange the scores into a DataFrame to be saved out to a CSV
    sys_weights = pd.DataFrame(index=list(set(mo['SystemName'])), columns=np.arange(2011, 2020))
    for sys in list(set(mo['SystemName'])):
        for season in np.arange(2011, 2020):
            if sys in weights_dict[season]:
                sys_weights.loc[sys, season] = weights_dict[season][sys]
                
    #Make sure all the weights are > 1 so we can use them in weighted averages and
    #the like
    sys_weights += abs(np.nanmin(sys_weights.astype(np.float64).values)) + 1
    
    sys_weights.to_csv('sys_weights.csv', index_label='SystemName')
    return sys_weights

'''
getRanks
Calculates overall rankings for each team for each season based on previously
calculated weightings. Requires sys_weights.csv to already be in the directory.

Params:
    df: DataFrame - getGames frame.
    files: Dict - getFiles dict.
    
Returns:
    wdf: DataFrame - df with T_Rank and O_Rank columns added.
'''
def getRanks(df, files):
    wdf = df.loc[df['Season'] >= 2011]
    wdf = wdf.sort_values('DayNum')
    weights = pd.read_csv('sys_weights.csv')
    wdf['T_Rank'] = 999
    wdf['O_Rank'] = 999
    mo = pd.read_csv(files['MMasseyOrdinals'])
    
    for idx, grp in tqdm(wdf.groupby(['Season', 'T_TeamID'])):
        #Sort values by GameID so they're in chronological order
        grp = grp.sort_values('GameID')
        
        #Get DataFrame of all the relevant rankings and weightings for that season
        ranks = mo.loc[np.logical_and(mo['Season'] == idx[0], mo['TeamID'] == idx[1])].merge(weights, on='SystemName').sort_values('RankingDayNum')
        
        #Calculate a weighted LOESS fit to the ranking curve. Gives more weight
        #to systems that are more accurate in that season
        wdf.loc[np.logical_and(wdf['Season'] == idx[0], 
                               wdf['T_TeamID'] == idx[1]), 'T_Rank'] = lowess(ranks['RankingDayNum'].values, 
                                       ranks['OrdinalRank'].values, 
                                       ranks[str(idx[0])].values, x0=grp['DayNum'], f=.25)
                                                                              
        #Apply these rankings to games where this team was the opponent
        wdf.loc[np.logical_and(wdf['Season'] == idx[0], 
                               wdf['O_TeamID'] == idx[1]), 'O_Rank'] = wdf.loc[np.logical_and(wdf['Season'] == idx[0], 
                               wdf['T_TeamID'] == idx[1]), 'T_Rank'].values
    
    #Save results out to a CSV so we don't have to wait for it to calculate these every time
    rank_df = wdf[['Season', 'GameID', 'DayNum', 'T_TeamID', 'O_TeamID', 'O_Rank', 'T_Rank']]
    rank_df.to_csv('rank_file.csv', index=False)
    return wdf

'''
addRanks
Adds T_Rank and O_Rank to frame, using already calculated rankings. Requires
rank_file.csv in directory.

Params:
    df: DataFrame - getGames frame.
    
Returns:
    wdf: DataFrame - df with T_Rank and O_Rank columns added.
'''
def addRanks(df):
    rdf = pd.read_csv('rank_file.csv')
    wdf = df.merge(rdf, on=['Season', 'GameID', 'DayNum', 'T_TeamID', 'O_TeamID'])
    return wdf

def calcElo(df):
    tids = list(set(df['T_TeamID'].values))
    elos = dict(zip(list(tids), [1500] * len(tids)))
    K = 20
    wdf = df[['GameID', 'T_TeamID', 'O_TeamID']]
    wdf['Elo'] = 1500
    
    for idx, day in tqdm(df.groupby(['GameID'])):
        gm = day.iloc[0]
        elo_diff = elos[gm['T_TeamID']] - elos[gm['O_TeamID']]
        mov = gm['T_Score'] - gm['O_Score']
        elo_shift = 1. / (10. ** (-elo_diff / 400.) + 1.)
        exp_margin = 7.5 + 0.006 * elo_diff
        final_elo_update = K * ((mov + 3.) ** 0.8) / exp_margin * (1 - elo_shift)
        elos[gm['T_TeamID']] += final_elo_update
        elos[gm['O_TeamID']] -= final_elo_update
        wdf.loc[np.logical_and(wdf['GameID'] == idx,
                               wdf['T_TeamID'] == gm['T_TeamID']), 'Elo'] = elos[gm['T_TeamID']]
        wdf.loc[np.logical_and(wdf['GameID'] == idx,
                               wdf['O_TeamID'] == gm['T_TeamID']), 'Elo'] = elos[gm['O_TeamID']]
        wdf.loc[np.logical_and(wdf['GameID'] == idx,
                               wdf['T_TeamID'] == gm['O_TeamID']), 'Elo'] = elos[gm['O_TeamID']]
        wdf.loc[np.logical_and(wdf['GameID'] == idx,
                               wdf['O_TeamID'] == gm['O_TeamID']), 'Elo'] = elos[gm['T_TeamID']]
    return wdf
            

'''
lowess
Locally Weighted Linear Regression that calculates a linear regression
to a small section of data then iterates over the whole dataset to create
a weighted linear regression.

Params:
    x: numpy array - (N,) array of independent data points
    y: numpy array - (N,) array of dependent data points
    w: numpy array - (N,) array of weights to use
    x0: numpy array - (M,) array of x points to interpolate to if wanted
    f: float - percentage of data to use in sub-regressions. Should be >= .1
                to avoid numerical instability or inability to converge.
    n_iter: int - number of smoothing iterations to perform.
    
Returns:
    yest: numpy array - (N,) or (M,) array of estimated points from the regression.
                            Size is determined by x, or x0 if available.
'''
def lowess(x, y, w, x0=None, f=.1, n_iter=3):
    n = len(x)
    r = int(np.ceil(f * n))
    yest = np.zeros(n)
    delta = np.ones(n)
    
    #Looping through all x-points
    for iteration in range(n_iter):
        for i in range(n):
            weights = np.zeros((n,))
            
            #get subset of weight points
            weights[max(0, i-r):min(i+r, n)] = w[max(0, i-r):min(i+r, n)]
            weights *= delta
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                        [np.sum(weights * x), np.sum(weights * x * x)]])
            theta = linalg.solve(A, b)
            yest[i] = theta[0] + theta[1] * x[i] 
        
        #use the residuals to modify weighting
        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2
        
    if x0 is None:
        return yest
    else:
        return np.interp(x0, x, yest)
    
    
        
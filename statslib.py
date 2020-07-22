'''

This is the statistics library that we'll be using.

'''

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.linalg import solve
from scipy.stats import hmean
import statsmodels.api as sm
from sklearn.preprocessing import RobustScaler

stat_names = ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 
              'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
id_cols = ['GameID', 'Season', 'DayNum', 'TID', 'OID']


'''
getFiles
Creates a dict of names for easy file access. Does this in an operating-system
agnostic way so we can work in Linux and Windows.

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
        ret = ret.rename(columns={'T_TeamID': 'TID', 'O_TeamID': 'OID'})
        return ret
    else:
        w = wdf_1.merge(ldf_1, on=['GameID', 'Season', 'DayNum', 'T_TeamID', 'O_TeamID'])
        l = wdf_2.merge(ldf_2, on=['GameID', 'Season', 'DayNum', 'T_TeamID', 'O_TeamID'])
        for col in w.columns:
            if col[-2:] == '_x':
                w = w.rename(columns={col: col[:-2]})
                l = l.rename(columns={col: col[:-2]})
            elif col[-2:] == '_y':
                w = w.drop(columns=[col])
                l = l.drop(columns=[col])
        w = w.rename(columns={'T_TeamID': 'TID', 'O_TeamID': 'OID'})
        l = l.rename(columns={'T_TeamID': 'TID', 'O_TeamID': 'OID'})
        return w, l
         
'''
normalizeToSeason
Modifies a DataFrame so each statistical columns has zero mean and a standard
deviation of one.

Params:
    df: DataFrame - frame to normalize.
    scaler: sklearn - one of the Scaler classes with a fit_transform method.
            None - uses mean and variance instead.
    
Returns:
    df: DataFrame - normalized frame.
'''
def normalizeToSeason(df, scaler=None):
    wdf = df.copy()
    if 'Season' in df.columns:
        for season, sdf in df.groupby(['Season']):
            if scaler is not None:
                wdf.loc[df['Season'] == season] = scaler.fit_transform(sdf)
            else:
                for col in sdf.columns:
                    if col not in ['GameID', 'Season', 'GLoc', 'DayNum', 'TID', 'OID', 'NumOT']:
                        wdf.loc[df['Season'] == season, col] = (sdf[col].values - sdf[col].mean()) / sdf[col].std()
    elif 'Season' in df.index.names:
        for season, sdf in df.groupby('Season'):
            if scaler is not None:
                wdf.loc[season] = scaler.fit_transform(sdf)
            else:
                for col in sdf.columns:
                    if col not in ['GameID', 'Season', 'GLoc', 'DayNum', 'TID', 'OID', 'NumOT']:
                        wdf.loc[season, col] = (sdf[col].values - sdf[col].mean()) / sdf[col].std()
    return wdf

'''
getDiffs
Get differences of each statistical column.

Params:
    df: DataFrame - getGames frame.

Returns:
    ret: DataFrame - frame with differences. Names are changed to reflect the
                        differencing.
'''
def getDiffs(df):
    ret = pd.DataFrame()
    for col in df.columns:
        if col[:2] == 'T_':
            ret[col[2:] + '_diff'] = df['T_' + col[2:]] - df['O_' + col[2:]]
        elif col[:2] == 'O_':
            continue
        else:
            ret[col] = df[col]
    return ret

'''
getTourneyStats
Adds various advanced statistical columns to a DataFrame based on tournament
play.

Params:
    tdf: DataFrame - getGames frame of tournament data.
    df: DataFrame - getGames frame.
    files: Dict - getFiles dict.
    
Returns:
    wdf: DataFrame - augmented frame.
'''
def getTourneyStats(tdf, df, files):
    wdf = pd.DataFrame(index=tdf.groupby(['Season', 'TID']).mean().index)
    df = df.sort_values('GameID')
    tdf = tdf.sort_values('GameID')
    seeds = pd.read_csv(files['MNCAATourneySeeds'])
    wdf['GameRank'] = 0
    for idx, team in tdf.groupby(['Season', 'TID']):
        if team.shape[0] < 6:
            loss = -1
            pts_added = np.exp(-.1 * abs(team['T_Score'].values[-1] - team['O_Score'].values[-1]))
        else:
            if team['T_Score'].values[-1] > team['O_Score'].values[-1]:
                loss = 0
                pts_added = 1 - np.exp(-.1 * abs(team['T_Score'].values[-1] - team['O_Score'].values[-1]))
            else:
                loss = -1
                pts_added = np.exp(-.1 * abs(team['T_Score'].values[-1] - team['O_Score'].values[-1]))
        
        wdf.loc[idx, 'GameRank'] = team.shape[0] + loss
        wdf.loc[idx, 'AdjGameRank'] = team.shape[0] + loss + pts_added
        sd = seeds.loc[np.logical_and(seeds['Season'] == idx[0],
                                      seeds['TeamID'] == idx[1]), 'Seed'].values[0][1:]
        sd = int(sd[:-1]) if len(sd) > 2 else int(sd)
        wdf.loc[idx, 'T_Seed'] = sd
        wdf.loc[idx, 'T_FinalElo'] = df.loc[np.logical_and(df['Season'] == idx[0],
                                                           df['TID'] == idx[1]),
                                            'T_Elo'].values[-1]
        wdf.loc[idx, 'T_FinalRank'] =df.loc[np.logical_and(df['Season'] == idx[0],
                                                           df['TID'] == idx[1]),
                                            'T_Rank'].values[-1]
    return wdf

'''
getStats
Adds various advanced statistical columns to a DataFrame.

Params:
    df: DataFrame - getGames frame.
    
Returns:
    wdf: DataFrame - augmented frame.
'''
def getStats(df):
    wdf = df[id_cols].copy()
    wdf['T_FG%'] = df['T_FGM'] / df['T_FGA']
    wdf['T_PPS'] = (df['T_Score'] - df['T_FTM']) / df['T_FGA']
    wdf['T_eFG%'] = (df['T_FGM'] + .5 * df['T_FGM3']) / df['T_FGA']
    wdf['T_TS%'] = df['T_Score'] / (2 * (df['T_FGA'] + .44 * df['T_FTA']))
    wdf['T_Econ'] = df['T_Ast'] + df['T_Stl'] - df['T_TO']
    wdf['T_Poss'] = .96 * (df['T_FGA'] - df['T_OR'] + df['T_TO'] + .44 * df['T_FTA'])
    wdf['T_OffRat'] = df['T_Score'] * 100 / wdf['T_Poss']
    wdf['T_R%'] = (df['T_OR'] + df['T_DR']) / (df['T_OR'] + df['T_DR'] + df['O_OR'] + df['O_DR'])
    wdf['T_Ast%'] = df['T_Ast'] / df['T_FGM']
    wdf['T_3Two%'] = df['T_FGA3'] / df['T_FGA']
    wdf['T_FT/A'] = df['T_FTA'] / df['T_FGA']
    wdf['T_FT%'] = df['T_FTM'] / df['T_FTA']
    wdf['T_TO%'] = df['T_TO'] / wdf['T_Poss']
     
    wdf['O_FG%'] = df['O_FGM'] / df['O_FGA']
    wdf['O_PPS'] = (df['O_Score'] - df['O_FTM']) / df['O_FGA']
    wdf['O_eFG%'] = (df['O_FGM'] + .5 * df['O_FGM3']) / df['O_FGA']
    wdf['O_TS%'] = df['O_Score'] / (2 * (df['O_FGA'] + .44 * df['O_FTA']))
    wdf['O_Econ'] = df['O_Ast'] + df['O_Stl'] - df['O_TO']
    wdf['O_Poss'] = .96 * (df['O_FGA'] - df['O_OR'] + df['O_TO'] + .44 * df['O_FTA'])
    wdf['O_OffRat'] = df['O_Score'] * 100 / wdf['O_Poss']
    wdf['O_R%'] = 1 - wdf['T_R%']
    wdf['O_Ast%'] = df['O_Ast'] / df['O_FGM']
    wdf['O_3Two%'] = df['O_FGA3'] / df['O_FGA']
    wdf['O_FT/A'] = df['O_FTA'] / df['O_FGA']
    wdf['O_FT%'] = df['O_FTM'] / df['O_FTA']
    wdf['O_TO%'] = df['O_TO'] / wdf['O_Poss']
    
    wdf['T_DefRat'] = wdf['O_OffRat']
    wdf['O_DefRat'] = wdf['T_OffRat']
    wdf['T_GameScore'] = 40 * wdf['T_eFG%'] + 20 * wdf['T_R%'] + 15 * wdf['T_FT/A'] + 25 - 25 * wdf['T_TO%']
    wdf['O_GameScore'] = 40 * wdf['O_eFG%'] + 20 * wdf['O_R%'] + 15 * wdf['O_FT/A'] + 25 - 25 * wdf['O_TO%']
    wdf['T_ProdPoss'] = wdf['T_Poss'] - df['T_TO'] - (df['T_FGA'] - df['T_FGM'] + .44 * df['T_FTM'])
    wdf['O_ProdPoss'] = wdf['O_Poss'] - df['O_TO'] - (df['O_FGA'] - df['O_FGM'] + .44 * df['O_FTM'])
    wdf['T_ProdPoss%'] = wdf['T_ProdPoss'] / wdf['T_Poss']
    wdf['O_ProdPoss%'] = wdf['O_ProdPoss'] / wdf['O_Poss']
    
    return wdf.fillna(0)

'''
getInfluenceStats
Calculates a few additional stats that take into account how the game played
changes a team's overall average.

Params:
    df: DataFrame - getGames frame.
    save: bool - determines whether to save a CSV of the result.
    recalc: bool - determines whether to calculate results or just use the CSV.
    
Returns:
    wdf: DataFrame - a frame with the calculated stats.
'''
def getInfluenceStats(df, save=True, recalc=False):
    if not recalc:
        return pd.read_csv('influence_stats.csv')
    wdf = df[id_cols].copy()
    for idx, team in tqdm(df.groupby(['Season', 'TID'])):
        opp_ids = np.logical_and(wdf['OID'] == idx[1], 
                               wdf['Season'] == idx[0])
        for col in team.columns:
            if 'T_' in col:
                wdf.loc[opp_ids, col + 'Shift'] = (team[col].values - np.mean(team[col])) / np.std(team[col])
    if save:
        wdf.to_csv('influence_stats.csv')
    return wdf
    

'''
getSeasonalStats
Calculates a weighted mean on each statistical column
and adds a few more stats that are valid for seasonal data.

Params:
    df: DataFrame - getGames frame, with columns added based on the weighting for the mean.
    strat: enum - This is a string that determines the averaging method we use.
        'rank': Weight by opponent's ranking
        'elo': weight by opponent's Elo
        'hmean': use harmonic mean
    save: bool - determines whether to save a CSV of the result.
    recalc: bool - determines whether to calculate results or just use the CSV.
        
    
Returns:
    wdf: DataFrame - frame with a single entry per team per season, with the means
                        and added stats for that team.
'''
def getSeasonalStats(df, strat='rank', seasonal_only=False):
    wdf = df.groupby(['Season', 'TID']).mean()
    avdf = df.copy()
    for id_col in ['Season', 'TID', 'GameID', 'GLoc', 'DayNum', 'OID']:
        if id_col in wdf.columns:
            wdf = wdf.drop(columns=[id_col])
            avdf = avdf.drop(columns=[id_col])
    dfapp = avdf.groupby(['Season', 'TID'])
    if not seasonal_only:
        if strat == 'rank':
            data = dfapp.apply(lambda x: np.average(x, axis=0, weights=400-x['O_Rank'].values))
        elif strat == 'elo':
            data = dfapp.apply(lambda x: np.average(x, axis=0, weights=x['O_Elo'].values))
        elif strat == 'hmean':
            data = dfapp.apply(lambda x: hmean(x + abs(x.min()) + .01, axis=0) - abs(x.min()) - .01)
        elif strat == 'mean':
            data = dfapp.mean()
        for idx, row in wdf.iterrows():
                wdf.loc[idx] = data.loc[idx]
    wdf['T_Win%'] = dfapp.apply(lambda x: sum(x['T_Score'] > x['O_Score']) / x.shape[0])
    wdf['T_PythWin%'] = dfapp.apply(lambda grp: sum(grp['T_Score']**13.91) / sum(grp['T_Score']**13.91 + grp['O_Score']**13.91))
    wdf['T_SoS'] = dfapp.apply(lambda grp: np.average(grp['O_Rank'], weights=grp['O_Elo']))
    if seasonal_only:
        wdf = wdf[['T_Win%', 'T_PythWin%', 'T_SoS']]
    return wdf

'''
getTeamStats
Gets a frame of only 'T_' stats, so we can look at a team's stats
without opponent involvement.

Params:
    df: DataFrame - getGames frame.
    av: bool - whether or not our frame is an averaged frame.
    
Returns:
    wdf: DataFrame - frame with all opponent stuff stripped.
'''
def getTeamStats(df, av=False):
    if av:
        wdf = pd.DataFrame(index=df.index)
    else:
        wdf = df[id_cols].drop(columns=['OID'])
    for col in df.columns:
        if col not in id_cols:
            if col[:2] == 'T_':
                wdf[col] = df[col]
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
system during each season. Scores are determined by the average value of 
a 2D Gaussian probability density function based on the difference between 
teams' ranking and the score of each game over the course of a season.
A higher number = a better weighting.

Params:
    df: DataFrame - getGames frame.
    files: Dict - getFiles dict.

Returns:
    sweights: DataFrame - frame of weightings for ranking systems. Index is
                                SystemName and the columns are the weightings
                                corresponding to each season.
    sprobs: DataFrame - frame showing percentage of correctly called
                                games for ranking systems. Same index
                                and columns as sweights.
'''
def getSystemWeights(df, files):
    #This should only need to be run once, since it saves its results out to a CSV
    mo = pd.read_csv(files['MMasseyOrdinals'])
    sweights = pd.DataFrame(data=np.nan, index=list(set(mo['SystemName'])), columns=list(set(df['Season'])))
    sprobs = pd.DataFrame(data=np.nan, index=list(set(mo['SystemName'])), columns=list(set(df['Season'])))
    
    #Chosen based on prior calculations. This is subjective, of course,
    #but should be pretty close to reality.
    gcov = np.linalg.pinv(np.array([[  226.274257  , -1141.89534807],
                                 [-1141.89534807, 12188.28247383]]))
    
    #Zero mean since the differences are symmetrical in a duplicated dataframe.
    gmu = np.array([0, 0])
    gaus2d = lambda x, y: np.exp(-(gcov[0, 0] * (x - gmu[0])**2 + 2 * gcov[0, 1] * \
                                   (x - gmu[0]) * (y - gmu[1]) + gcov[1, 1] * (y - gmu[1])**2))
    diffs = getDiffs(df).sort_values('DayNum')
    for season in sweights.columns:
        wdf_diffs = diffs.loc[diffs['Season'] == season]
        wdf_diffs['Rank_diff'] = 999
        mo_season = mo.loc[mo['Season'] == season]
        '''
        WEIGHTING SYSTEM FOR MASSEY ORDINALS
        '''
        for idx, sys in tqdm(mo_season.groupby(['SystemName'])):
            for tid, team in sys.groupby(['TeamID']):
                if team.shape[0] < 2:
                    ranks = team['OrdinalRank'].values[0] * np.ones((wdf_diffs.loc[wdf_diffs['TID'] == tid].shape[0],))
                else:
                    #Interpolate between system outputs to get a team's rank at the time of an actual game
                    #This can be improved on, probably. Only using linear interpolation.
                    ranks = np.interp(wdf_diffs.loc[wdf_diffs['TID'] == tid, 'DayNum'], 
                                      team['RankingDayNum'], team['OrdinalRank'], left=team['OrdinalRank'].values[0])
                wdf_diffs.loc[wdf_diffs['TID'] == tid, 'T_Rank'] = ranks
                wdf_diffs.loc[wdf_diffs['OID'] == tid, 'O_Rank'] = ranks
            rdiff = wdf_diffs['T_Rank'] - wdf_diffs['O_Rank']
            scores = gaus2d(wdf_diffs['Score_diff'], rdiff)
            sweights.loc[idx, season] = sum(scores[np.logical_or(np.logical_and(rdiff < 0, 
                                                                  wdf_diffs['Score_diff'] > 0),
                                                                 np.logical_and(rdiff > 0, 
                                                                  wdf_diffs['Score_diff'] < 0))]) / len(scores)
            sprobs.loc[idx, season] = sum(np.logical_or(np.logical_and(rdiff < 0, 
                                                         wdf_diffs['Score_diff'] > 0),
                                                        np.logical_and(rdiff > 0, 
                                                         wdf_diffs['Score_diff'] < 0))) / len(scores)
    
    sweights.to_csv('sys_weights.csv', index_label='SystemName')
    return sweights, sprobs

'''
getRanks
Calculates overall rankings for each team for each season based on previously
calculated weightings, using a LOESS fit to consolidate ranking systems. 
Requires sys_weights.csv to already be in the directory.

Params:
    df: DataFrame - getGames frame.
    files: Dict - getFiles dict.
    
Returns:
    wdf: DataFrame - df with T_Rank and O_Rank columns added.
'''
def getRanks(df, files):
    wdf = df.copy()
    wdf = wdf.sort_values('DayNum')
    weights = pd.read_csv('sys_weights.csv')
    wdf['T_Rank'] = 999
    wdf['O_Rank'] = 999
    mo = pd.read_csv(files['MMasseyOrdinals'])
    
    for idx, grp in tqdm(wdf.groupby(['Season', 'TID'])):
        #Sort values by GameID so they're in chronological order
        grp = grp.sort_values('GameID')
        
        #Get DataFrame of all the relevant rankings and weightings for that season
        ranks = mo.loc[np.logical_and(mo['Season'] == idx[0], mo['TeamID'] == idx[1])].merge(weights, on='SystemName').sort_values('RankingDayNum')

        #Calculate a weighted LOESS fit to the ranking curve. Gives more weight
        #to systems that are more accurate in that season
        wdf.loc[np.logical_and(wdf['Season'] == idx[0], 
                               wdf['TID'] == idx[1]), 'T_Rank'] = lowess(ranks['RankingDayNum'].values, 
                                       ranks['OrdinalRank'].values, 
                                       ranks[str(idx[0])].values, x0=grp['DayNum'], f=.25)
                                                                              
        #Apply these rankings to games where this team was the opponent
        wdf.loc[np.logical_and(wdf['Season'] == idx[0], 
                               wdf['OID'] == idx[1]), 'O_Rank'] = wdf.loc[np.logical_and(wdf['Season'] == idx[0], 
                               wdf['TID'] == idx[1]), 'T_Rank'].values
    
    #Save results out to a CSV so we don't have to wait for it to calculate these every time
    rank_df = wdf[['Season', 'GameID', 'DayNum', 'TID', 'OID', 'O_Rank', 'T_Rank']]
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
    wdf = df.merge(rdf, on=['Season', 'GameID', 'DayNum', 'TID', 'OID'])
    return wdf

'''
calcElos
Runs through the entire frame and calculates Elo scores for each team, adding
them to a separate dataframe. Saves the results to a CSV for ease of access later.
NOTE: Use one of the split=True frames to do this, it won't work as expected
otherwise.

Params:
    df: DataFrame - getGames frame.
    K: float - Memory parameter, determines the exponential decay in the calculation.
                35 is chosen based on percentage of correct matchup predictions.
    margin: float - determines the 'margin of victory' parameter that 538 puts in to make Elo better.
                4.5 is chosen based on percentage of correct matchup predictions.
    
Returns:
    wdf: DataFrame - frame with GameID, TID, OID, T_Elo, and O_Elo columns.
'''
def calcElo(df, K=35, margin=4.5):
    tids = list(set(df['TID'].values))
    elos = dict(zip(list(tids), [1500] * len(tids)))
    wdf = df[['GameID', 'TID', 'OID']].copy()
    wdf.loc[:, 'T_Elo'] = 1500
    wdf.loc[:, 'O_Elo'] = 1500
    season = df['Season'].min()
    for idx, gm in tqdm(df.iterrows()):
        #First, regression to the mean if the season is new
        if season != gm['Season']:
            for key in elos:
                elos[key] = .25 * elos[key] + .75 * 1500
            season = gm['Season']
        #Elo calculation stolen from 538
        elo_diff = elos[gm['TID']] + (gm['GLoc'] * 100) - elos[gm['OID']]
        mov = gm['T_Score'] - gm['O_Score']
        elo_shift = 1. / (10. ** (-elo_diff / 400.) + 1.)
        exp_margin = margin + 0.006 * elo_diff
        final_elo_update = K * ((mov + 3.) ** 0.8) / exp_margin * (1 - elo_shift)
        elos[gm['TID']] += final_elo_update
        elos[gm['OID']] -= final_elo_update
        wdf.loc[idx, ['T_Elo', 'O_Elo']] = [elos[gm['TID']], elos[gm['OID']]]
    wdf2 = wdf.copy()
    wdf2['TID'], wdf2['OID'] = wdf['OID'], wdf['TID']
    wdf2['T_Elo'], wdf2['O_Elo'] = wdf['O_Elo'], wdf['T_Elo']
    wdf = wdf.append(wdf2, ignore_index=True)
    wdf.to_csv('elo_file.csv', index=False)
    return wdf

'''
addElos
Adds T_Elo and O_Elo to frame, using already calculated elo values. Requires
elo_file.csv in directory. Modifies in-place.

Params:
    df: DataFrame - getGames frame.
    
Returns:
    df: DataFrame - df with T_Rank and O_Rank columns added.
'''
def addElos(df):
    rdf = pd.read_csv('elo_file.csv')
    df = df.merge(rdf, on=['GameID', 'TID', 'OID'])
    return df

'''
joinFrame
Merges two getGames frames on their id columns.

Params:
    df1: DataFrame - getGames frame.
    df2: DataFrame - getGames frame.
    
Returns:
    df: DataFrame - df1 and df2, merged on id columns.
'''
def joinFrame(df1, df2):
    return df1.merge(df2, on=['GameID', 'Season', 'DayNum', 'TID', 'OID'])
            

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
            theta = solve(A, b)
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
    
    
        
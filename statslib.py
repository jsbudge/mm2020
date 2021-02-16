'''

This is the statistics library that we'll be using.

'''

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.linalg import solve
from scipy.interpolate import CubicSpline
from scipy.stats import hmean, iqr
import statsmodels.api as sm
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import HuberRegressor, Lars, ElasticNet, Lasso, SGDRegressor, TheilSenRegressor, \
    ARDRegression, LassoLars
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from itertools import combinations, permutations

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
    split: bool - Determines whether to return a single, double size DataFrame or
                    a tuple of DataFrames arranged by winners and losers
                
Returns:
    split = True
        w: DataFrame - frame arranged as T_ = winners of games
        l: DataFrame - frame arranged as T_ = losers of games
    split = False
        ret: DataFrame - frame of all games, double the length of the original CSV
'''
def getGames(season=None, split=False, tourney=False):
    if not tourney:
        df = pd.read_csv('./data/MRegularSeasonDetailedResults.csv')
    else:
        df = pd.read_csv('./data/MNCAATourneyDetailedResults.csv')
    if season is not None:
        df = df.loc[df['Season'] == season]
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
    
"""
NOT COMPLETE
"""
def addRegressionStats(df, gr):
    sts = df.copy()
    lms = [('RF', RandomForestRegressor(n_estimators=500)),
       ('SGD', SGDRegressor()),
       ('ARD', ARDRegression()),
       ('NN', MLPRegressor(max_iter=1000)),
       ('Theil', TheilSenRegressor())]
    for lm in lms:
        sgd = make_pipeline(StandardScaler(), 
                            lm[1]).fit(sts, gr)
        sts[lm[0]] = sgd.predict(sts)
    return sts
         
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
    for season, sdf in df.groupby('Season'):
        if scaler is not None:
            wdf.loc(axis=0)[:, season, :, :] = scaler.fit_transform(sdf)
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
            try:
                ret[col[2:] + '_diff'] = df['T_' + col[2:]] - df['O_' + col[2:]]
            except:
                ret[col[2:]] = df['T_' + col[2:]]
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
    
Returns:
    wdf: DataFrame - augmented frame.
'''
def getTourneyStats(tdf, df):
    wdf = pd.DataFrame(index=tdf.groupby(['Season', 'TID']).mean().index)
    df = df.sort_values('GameID')
    tdf = tdf.sort_values('GameID')
    seeds = pd.read_csv('./data/MNCAATourneySeeds.csv')
    for idx, team in tdf.groupby(['Season', 'TID']):
        wdf.loc[idx, 'T_Momentum'] = np.mean(np.gradient(df.loc(axis=0)[:, idx[0], idx[1], :]['T_Elo'].values)[-3:])
        wdf.loc[idx, 'T_RankNoise'] = np.std(df.loc(axis=0)[:, idx[0], idx[1], :]['T_Rank'].values)
        sd = seeds.loc[np.logical_and(seeds['Season'] == idx[0],
                                      seeds['TeamID'] == idx[1]), 'Seed'].values[0][1:]
        sd = int(sd[:-1]) if len(sd) > 2 else int(sd)
        wdf.loc[idx, 'T_Seed'] = sd
        wdf.loc[idx, 'T_FinalElo'] = df.loc(axis=0)[:, idx[0], idx[1], :]['T_Elo'].values[-1]
        wdf.loc[idx, 'T_FinalRank'] = df.loc(axis=0)[:, idx[0], idx[1], :]['T_Rank'].values[-1]
        wdf.loc[idx, 'T_RoundRank'] = sum(team['T_Score'] > team['O_Score'])
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
    wdf['T_FG3%'] = df['T_FGM3'] / df['T_FGA3']
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
    wdf['T_ExtraPoss'] = df['T_OR'] + df['T_Stl'] + df['O_PF']
     
    wdf['O_FG%'] = df['O_FGM'] / df['O_FGA']
    wdf['O_FG3%'] = df['O_FGM3'] / df['O_FGA3']
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
    wdf['O_ExtraPoss'] = df['O_OR'] + df['O_Stl'] + df['T_PF']
    
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
def getInfluenceStats(df, save=True, recalc=False, norm=False):
    if not recalc:
        return pd.read_csv('./data/influence_stats.csv')
    scale_df = df.groupby(['Season', 'TID']).apply(lambda x: (x - x.mean()) / x.std())
    scale_std = df.groupby(['Season']).std()
    inf_df = scale_df.groupby(['Season', 'OID']).mean()
    for idx, grp in inf_df.groupby(['Season']):
        inf_df.loc[grp.index] = grp * scale_std.loc[idx]
    inf_df = inf_df.drop(columns=[col for col in inf_df.columns if 'O_' in col])
    inf_df = inf_df.reset_index().rename(columns={'OID': 'TID'}).set_index(['Season', 'TID'])
    if norm:
        inf_df = normalizeToSeason(inf_df)
    inf_df.columns = [col + 'Inf' for col in inf_df.columns]
    if save:
        inf_df.to_csv('./data/influence_stats.csv')
    return inf_df
    

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
        elif strat == 'relelo':
            data = dfapp.apply(lambda x: np.average(x, axis=0, weights= .1 / abs(x['T_Elo'] - x['O_Elo']).values**(1/4)))
        elif strat == 'gausselo':
            data = dfapp.apply(lambda x: np.average(x, axis=0, weights=1 / (100 * np.sqrt(2 * np.pi)) * np.exp(-.5 * ((x['T_Elo'] - x['O_Elo']).values / 100)**2)))
        elif strat == 'mest':
            data = dfapp.apply(lambda x: np.average(x, axis=0, weights=1 / (iqr(x, axis=0) * np.sqrt(2 * np.pi)) * np.exp(-.5 * ((x - x.median()).values / iqr(x, axis=0))**2)))
        elif strat == 'recent':
            data = dfapp.apply(lambda x: np.average(x, axis=0, weights=np.arange(x.shape[0])**2))
        elif strat == 'mean':
            data = dfapp.mean()
        for idx, row in wdf.iterrows():
                wdf.loc[idx] = data.loc[idx]
    wdf['T_Win%'] = dfapp.apply(lambda x: sum(x['T_Score'] > x['O_Score']) / x.shape[0])
    wdf['T_PythWin%'] = dfapp.apply(lambda grp: sum(grp['T_Score']**13.91) / sum(grp['T_Score']**13.91 + grp['O_Score']**13.91))
    wdf['T_SoS'] = dfapp.apply(lambda grp: np.average(400-grp['O_Rank'], weights=grp['O_Elo']) / 4)
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
    wdf = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col[:2] == 'T_':
            wdf[col] = df[col]
    return wdf
        
'''
loadTeamNames
Look up a team and/or ID quickly and easily.
    
Returns:
    ret: Dict - dict: index is TeamID or TeamName, and elements have the TeamName
                        and TeamID, respectively.
'''
def loadTeamNames():
    df = pd.read_csv('./data/MTeams.csv')
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

Returns:
    sweights: DataFrame - frame of weightings for ranking systems. Index is
                                SystemName and the columns are the weightings
                                corresponding to each season.
    sprobs: DataFrame - frame showing percentage of correctly called
                                games for ranking systems. Same index
                                and columns as sweights.
'''
def calcSystemWeights():
    #This should only need to be run once, since it saves its results out to a CSV
    df = getGames('./data/MRegularSeasonDetailedResults.csv', split=False)
    mo = pd.read_csv('./data/MMasseyOrdinals.csv')
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
                    #Linear interpolation
                    # ranks = np.interp(wdf_diffs.loc[wdf_diffs['TID'] == tid, 'DayNum'], 
                    #                   team['RankingDayNum'], team['OrdinalRank'], left=team['OrdinalRank'].values[0])
                    #Cubic spline
                    ranks = CubicSpline(team['RankingDayNum'], team['OrdinalRank'])(wdf_diffs.loc[wdf_diffs['TID'] == tid, 'DayNum'].values)
                    ranks[wdf_diffs.loc[wdf_diffs['TID'] == tid, 'DayNum'].values < team['RankingDayNum'].values[0]] = team['OrdinalRank'].values[0]
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
    
    sweights.to_csv('./data/sys_weights.csv', index_label='SystemName')
    return sweights, sprobs

'''
getRanks
Calculates overall rankings for each team for each season based on previously
calculated weightings, using a LOESS fit to consolidate ranking systems. 
Requires sys_weights.csv to already be in the directory.

Params:
    df: DataFrame - getGames frame.
    
Returns:
    wdf: DataFrame - df with T_Rank and O_Rank columns added.
'''
def getRanks(df):
    wdf = df.copy()
    wdf = wdf.sort_values('DayNum')
    weights = pd.read_csv('./data/sys_weights.csv')
    wdf['T_Rank'] = 999
    wdf['O_Rank'] = 999
    mo = pd.read_csv('./data/MMasseyOrdinals.csv')
    
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
    rank_df.to_csv('./data/rank_file.csv', index=False)
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
    rdf = pd.read_csv('./data/rank_file.csv')
    wdf = df.merge(rdf, on=['Season', 'GameID', 'DayNum', 'TID', 'OID'])
    return wdf

'''
calcElos
Runs through the entire frame and calculates Elo scores for each team, adding
them to a separate dataframe. Saves the results to a CSV for ease of access later.
NOTE: Use one of the split=True frames to do this, it won't work as expected
otherwise.

Params:
    K: float - Memory parameter, determines the exponential decay in the calculation.
                35 is chosen based on percentage of correct matchup predictions.
    margin: float - determines the 'margin of victory' parameter that 538 puts in to make Elo better.
                4.5 is chosen based on percentage of correct matchup predictions.
    
Returns:
    wdf: DataFrame - frame with GameID, TID, OID, T_Elo, and O_Elo columns.
'''
def calcElo(K=35, margin=4.5):
    df = pd.read_csv('./data/MRegularSeasonCompactResults.csv')
    tids = list(set(df['WTeamID'].values) & set(df['LTeamID'].values))
    elos = dict(zip(list(tids), [1500] * len(tids)))
    wdf = df[['Season', 'WTeamID', 'LTeamID', 'WLoc', 'DayNum']].copy().rename(columns={'WTeamID': 'TID', 'LTeamID': 'OID'})
    wdf = wdf.loc[wdf['Season'] >= 2000]
    wdf.loc[:, 'T_Elo'] = 1500
    wdf.loc[:, 'O_Elo'] = 1500
    wdf['ScoreDiff'] = df['WScore'] - df['LScore']
    season = df['Season'].min()
    for idx, gm in tqdm(wdf.iterrows()):
        #First, regression to the mean if the season is new
        if season != gm['Season']:
            for key in elos:
                elos[key] = .15 * elos[key] + .85 * 1500
            season = gm['Season']
        #Elo calculation stolen from 538
        hc_adv = 100 if gm['WLoc'] == 'H' else -100 if gm['WLoc'] == 'A' else 0
        elo_diff = elos[gm['TID']] + hc_adv - elos[gm['OID']]
        elo_shift = 1. / (10. ** (-elo_diff / 400.) + 1.)
        exp_margin = margin + 0.006 * elo_diff
        final_elo_update = K * ((gm['ScoreDiff'] + 3.) ** 0.8) / exp_margin * (1 - elo_shift)
        elos[gm['TID']] += final_elo_update
        elos[gm['OID']] -= final_elo_update
        wdf.loc[idx, ['T_Elo', 'O_Elo']] = [elos[gm['TID']], elos[gm['OID']]]
    wdf2 = wdf.copy()
    wdf2['TID'], wdf2['OID'] = wdf['OID'], wdf['TID']
    wdf2['T_Elo'], wdf2['O_Elo'] = wdf['O_Elo'], wdf['T_Elo']
    wdf = wdf.append(wdf2, ignore_index=True)
    wdf = wdf.drop(columns=['WLoc', 'ScoreDiff'])
    wdf.to_csv('./data/elo_file.csv', index=False)
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
    rdf = pd.read_csv('./data/elo_file.csv')
    df = df.merge(rdf, on=['Season', 'TID', 'OID', 'DayNum'])
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
arrangeFrame
arranges games into frames for stats, winners, and dayNum.

Params:
    season: int - If we want a particular season. Leave as None to get everything.
    scaling: sklearn.preprocessing scaler - A scaler to normalize the stats with.
                Leave this as None to just get the stats as they are.
    noraw: bool - Set this as True to only get advanced stats, and not things
                like points and rebounds.
    noinfluence: bool - Set this as True to drop influence stats.

Returns:
    ts: DataFrame - Frame with stats.
    ty: DataFrame - Frame (technically a Series) of True/False based on if the
                the TID team won the game.
    tsdays: DataFrame - Frame (Series) of DayNum values.
    All frames use GameID, Season, TID, OID as their indexes.
'''

def arrangeFrame(season=None, scaling=None, noraw=False, noinfluence=False,
                 split=False):
    if split:
        ts = getGames(season=season, split=True)[0].drop(columns=['NumOT', 'GLoc'])
    else:
        ts = getGames(season=season).drop(columns=['NumOT', 'GLoc'])
    ty = ts['T_Score'] > ts['O_Score'] - 0
    if noraw:
        ts = joinFrame(ts[['GameID', 'Season', 'DayNum', 'TID', 'OID', 'T_Score', 'O_Score']],
                          getStats(ts))
    else:
        ts = joinFrame(ts, getStats(ts))
    ts = addRanks(ts)
    ts = addElos(ts)
    ts = ts.set_index(['GameID', 'Season', 'TID', 'OID'])
    tsdays = ts['DayNum']
    if not noinfluence:
        ts = joinFrame(ts, getInfluenceStats(ts)).set_index(['GameID', 'Season', 'TID', 'OID'])
        ts = ts.drop(columns=['DayNum', 'Unnamed: 0'])
    if scaling is not None:
        ts = normalizeToSeason(ts, scaler=scaling)
    return ts, ty, tsdays

'''
arrangeTourneyGames
arranges tournament games into frames for stats, winners, and dayNum.

Params:
    noraw: bool - Set this as True to only get advanced stats, and not things
                like points and rebounds.

Returns:
    tts: DataFrame - Frame with stats.
    tty: DataFrame - Frame (technically a Series) of True/False based on if the
                the TID team won the game.
    ttsdays: DataFrame - Frame (Series) of DayNum values.
    All frames use GameID, Season, TID, OID as their indexes.
'''
def arrangeTourneyGames(noraw=False):
    tts = getGames(tourney=True).drop(columns=['NumOT', 'GLoc'])
    tty = tts['T_Score'] > tts['O_Score'] - 0
    if noraw:
        tts = joinFrame(tts[['GameID', 'Season', 'DayNum', 'TID', 'OID', 'T_Score', 'O_Score']],
                           getStats(tts)).set_index(['GameID', 'Season', 'TID', 'OID'])
    else:
        tts = joinFrame(tts, getStats(tts)).set_index(['GameID', 'Season', 'TID', 'OID'])
    ttsdays = tts['DayNum']
    tts = tts.drop(columns=['DayNum'])
    tty.index = tts.index
    return tts, tty, ttsdays

'''
getMatches
Arranges team feature vectors into games for easy training.

Params:
    gids: DataFrame - Frame of games you want to use feature vectors for.
                This only uses the index of this frame.
    team_feats: DataFrame - Frame of feature vectors for each team. Should have
                index of [Season, TID].
    season: int - set this to get a particular season of the frame passed to
                gids.
    diff: bool - set this to True if you want differences between teams, not
                a frame with both teams' stats.

Returns:
    fdf: DataFrame - Frame with team_feats columns for both teams, using gids'
                index.
'''
        
def getMatches(gids, team_feats, season=None, diff=False):
    if season is not None:
        g = gids.loc(axis=0)[:, season, :, :]
    else:
        g = gids.copy()
    ids = ['GameID', 'Season', 'TID', 'OID']
    gsc = g.reset_index()[ids]
    g1 = gsc.merge(team_feats, on=['Season', 'TID']).set_index(ids)
    g2 = gsc.merge(team_feats, left_on=['Season', 'OID'],
                   right_on=['Season', 'TID']).set_index(ids)
    fdf = g1 - g2 if diff else g1.merge(g2, on=ids)
    return fdf
    
        
def getAllMatches(team_feats, season, diff=False):
    sd = pd.read_csv('./data/MNCAATourneySeeds.csv')
    sd = sd.loc[sd['Season'] == season]['TeamID'].values
    teams = list(set(sd))
    matches = [[x, y] for (x, y) in permutations(teams, 2)]
    poss_games = pd.DataFrame(data=matches, columns=['TID', 'OID'])
    poss_games['Season'] = season; poss_games['GameID'] = np.arange(poss_games.shape[0])
    gsc = poss_games.set_index(['GameID', 'Season'])
    g1 = gsc.merge(team_feats, on=['Season', 'TID'], 
                   right_index=True).sort_index()
    g1 = g1.reset_index().set_index(['GameID', 'Season', 'TID', 'OID'])
    g2 = gsc.merge(team_feats, left_on=['Season', 'OID'],
                   right_on=['Season', 'TID'],
                   right_index=True).sort_index()
    g2 = g2.reset_index().set_index(['GameID', 'Season', 'TID', 'OID'])
    fdf = g1 - g2 if diff else g1.merge(g2, on=['GameID', 'Season', 'TID', 'OID'])
    return fdf

def merge(*args):
    df = None
    for d in args:
        if df is None:
            df = d
        else:
            df = df.merge(d, left_index=True, right_index=True)
    return df
            

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
        
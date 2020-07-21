#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 11:07:54 2020

@author: jeff

Library for holding all the plotting tools I want.
Mostly because it's cool, not adding much to the analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import pandas as pd
from statslib import loadTeamNames, getDiffs, normalizeToSeason
import itertools
from sklearn.feature_selection import SelectPercentile, mutual_info_classif, f_classif

'''
PlotGenerator
Class for holding data, with additional methods for plotting it in a more complex
way than when we just want something quick and dirty from the command line.

Params:
    df: DataFrame - getGames frame to hold plotting data
    seasonal_df: DataFrame - holds seasonal data for comparison to other teams.
    files: Dict - getFiles dict for extra files needed.
    season: int - if we want to just look at one season for simplicity or speed.
'''
class PlotGenerator(object):

    def __init__(self, df, seasonal_df, files, season = 2018):
        self.df = df.sort_values('GameID')
        self.sdf = seasonal_df
        self.default_season = season
        self.files = files
        self.names = loadTeamNames(files)
                
    def showTeamOverall(self, tid, season=None):
        season = self.default_season if season is None else season
        if type(tid) == str:
            tid = self.names[tid]
        tdf = self.df.loc[self.df['TID'] == tid]
        tdf = tdf.loc[tdf['Season'] == season]
        sdata = self.sdf.loc(axis=0)[season, :]
        sdata = (sdata - sdata.mean()) / sdata.std()
        f = plt.figure('{}: '.format(tid) + self.names[tid])
        gd = gridspec.GridSpec(3, 2, figure=f)
        f.add_subplot(gd[1, :])
        sns.regplot(data=self.df.loc[self.df['Season'] == season], x='DayNum', y='T_Elo', order=4, scatter_kws={'s': 1})
        plt.plot(tdf['DayNum'], tdf['T_Elo'])
        f.add_subplot(gd[2, :])
        plt.plot(tdf['DayNum'], tdf['T_Rank'])
        ax1 = f.add_subplot(gd[0, 0], projection='polar')
        plt.title('Offense')
        off_stats = ['T_eFG%', 'T_TS%', 'T_Ast', 'T_Ast%', 'T_PPS', 'T_OffRat', 'T_FT/A', 'T_Econ', 'T_3Two%']
        theta = np.arange(len(off_stats)) * 2 * np.pi / len(off_stats)
        theta_team = np.arange(len(off_stats)+1) * 2 * np.pi / len(off_stats)
        r = np.concatenate((sdata.loc[(season, tid), off_stats].values, [sdata.loc[(season, tid), off_stats[0]]]))
        rs = sdata[off_stats].mean().values
        plt.polar(theta, rs)
        plt.fill(theta, rs)
        plt.polar(theta_team, r)
        ax1.set_xticks(theta)
        ax1.set_xticklabels(off_stats)
        ax1 = f.add_subplot(gd[0, 1], projection='polar')
        plt.title('Defense')
        def_stats = ['T_DefRat', 'T_R%', 'T_OR', 'T_DR', 'T_Stl', 'T_Blk', 'T_TO', 'T_PF']
        theta = np.arange(len(def_stats)) * 2 * np.pi / len(def_stats)
        theta_team = np.arange(len(def_stats)+1) * 2 * np.pi / len(def_stats)
        r = np.concatenate((sdata.loc[(season, tid), def_stats].values, [sdata.loc[(season, tid), def_stats[0]]]))
        rs = sdata[def_stats].mean().values
        plt.polar(theta, rs)
        plt.fill(theta, rs)
        plt.polar(theta_team, r)
        ax1.set_xticks(theta)
        ax1.set_xticklabels(def_stats)
        
def showCorrs(df1, df2):
    disp = pd.DataFrame(index=df1.columns, columns=df2.columns)
    for col in df2.columns:
        disp[col] = abs(df1.corrwith(df2[col]))
    plt.figure('Abs Corrs')
    sns.heatmap(disp)
        
def showStat(df, stat, season, tid):
    tdf = df.loc[np.logical_and(df['Season'] == season,
                                   df['TID'] == tid)].sort_values('DayNum')
    plt.figure(stat)
    sns.scatterplot(data=tdf, x='DayNum', y=stat)
    plt.plot(tdf['DayNum'], 
             tdf[stat])
    
def showScore(X, y):
    disp = pd.DataFrame(index=X.columns, columns=['ANOVA_F', 'MINFO'])
    disp['ANOVA_F'] = f_classif(X, y)[1]
    disp['MINFO'] = mutual_info_classif(X, y)
    disp = disp.T.reset_index()
    plt.figure()
    sns.barplot(data=disp.melt(id_vars=['index']), x='variable', y='value', hue='index', errwidth=0)
    plt.xticks(rotation=45)
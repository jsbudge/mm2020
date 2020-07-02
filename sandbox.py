#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:31:29 2020

@author: jeff

Play script

Stuff here gets added to statslib later.
"""

import numpy as np
import pandas as pd
import statslib as st
import seaborn as sns
from scipy.interpolate import CubicSpline
from tqdm import tqdm



files = st.getFiles()

wdf = st.getGames(files['MRegularSeasonDetailedResults'], True, False)[0]
wdf2019 = st.getGames(files['MRegularSeasonDetailedResults'], False, True)
wdf2019 = wdf2019.loc[wdf2019['Season'] == 2019]
mo = pd.read_csv(files['MMasseyOrdinals'])
mo2019 = mo.loc[mo['Season'] == 2019]

ranksys = {}

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
                                                                                                                       
            

wdf_diffs = st.getDiffs(wdf.loc[wdf['Season'] == 2019])
for idx, row in tqdm(wdf_diffs.iterrows()):
    for sys in ranksys:
        try:
            ranksys[sys]['rankscore'] -= \
                (ranksys[sys][row['1_TeamID']][wdf2019.loc[wdf2019['TeamID'] == row['1_TeamID'], 'DayNum'] == row['DayNum']][0] \
                 - ranksys[sys][row['2_TeamID']][wdf2019.loc[wdf2019['TeamID'] == row['2_TeamID'], 'DayNum'] == row['DayNum']][0]) / wdf_diffs.shape[0] * row['Score_diff'] / 39
        except:
            ranksys[sys]['rankscore'] -= 0
            
wdf2019['Rank'] = 999
for idx, row in tqdm(wdf2019.iterrows()):
    try:
        wghts = np.array([ranksys[sys]['rankscore'] for sys in ranksys if row['TeamID'] in ranksys[sys]])
        rnks = np.array([ranksys[sys][row['TeamID']][wdf2019.loc[wdf2019['TeamID'] == row['TeamID'], 'DayNum'] == row['DayNum']][0] for sys in ranksys if row['TeamID'] in ranksys[sys]])
        wdf2019.loc[idx, 'Rank'] = np.average(rnks, weights=wghts)
    except:
        wdf2019.loc[idx, 'Rank'] = 999
    


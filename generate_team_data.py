#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:15:19 2021

@author: jeff

Grabs internet team data. Should be faster than the pplayer stuff.
"""

from sportsreference.ncaab.roster import Player
from sportsreference.ncaab.teams import Teams, Team
import pandas as pd
from tqdm import tqdm
import statslib as st
import eventlib as ev
import numpy as np
from difflib import get_close_matches

add_to_existing = True
pdf = ev.getTeamRosters()
#Grab the data from sportsreference
seas_map = {'2011-12': 2012,
            '2012-13': 2013, '2013-14': 2014,
            '2014-15': 2015, '2015-16': 2016,
                                    '2016-17': 2017, '2017-18': 2018,
                                    '2018-19': 2019, '2019-20': 2020,
                                    '2020-21': 2021, '2021-2022': 2022}
play_df = pd.DataFrame()
for season in [2021]:
    tm = Teams(season)
    for team in tqdm(tm):
        try:
            df = team.dataframe
            df['Season'] = season
            play_df = play_df.append(df)
        except:
            print(team.name)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 11:26:59 2020

@author: jeff

Initializes all the CSV files.
"""

import numpy as np
import pandas as pd
import statslib as st
import framelib as fl

df = st.getGames(files['MRegularSeasonDetailedResults'])
print('Calculating Elo...')
elos = st.calcElo()
print('Calculating system weights...')
weights = st.calcSystemWeights()
print('Calculating rankings...')
ranks = st.getRanks(df)
print('Calculating influence stats...')
inf_df = st.getInfluenceStats(df, True)
final_df = fl.arrangeFrame()

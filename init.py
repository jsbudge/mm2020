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

files = st.getFiles()
df = st.getGames(files['MRegularSeasonDetailedResults'])
print('Calculating Elo...')
elos = st.calcElo(files)
print('Calculating system weights...')
weights = st.calcSystemWeights(files)
print('Calculating rankings...')
ranks = st.getRanks(df, files)
print('Calculating influence stats...')
df = st.joinFrame(df, st.getStats(df))
influence = st.getInfluenceStats(df, recalc=True)

final_df = fl.arrangeFrame(files)

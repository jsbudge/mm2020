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
import internetlib as il

df = st.getGames(split=False)
print('Calculating Elo...')
elos = st.calcElo()
print('Calculating system weights...')
weights = st.calcSystemWeights()
print('Calculating rankings...')
ranks = st.getRanks()
print('Calculating influence stats...')
sdf = st.arrangeFrame(scaling=None, noinfluence=True)[0]
inf_df = st.getInfluenceStats(sdf, recalc=True)
print('Grabbing player data from internet...')
#This one will take a while
il.getPlayerData(add_to_existing=False)

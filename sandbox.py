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



files = st.getFiles()

wdf = st.getGames(files['MRegularSeasonDetailedResults'], True, True)

wdf = st.normalizeToSeason(wdf)
sns.kdeplot(wdf['1_Score'])

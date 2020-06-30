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
from statslib import getFiles, getGames, normalizeToSeason
import seaborn as sns



files = getFiles()

wdf = getGames(files['MRegularSeasonDetailedResults'], False, True)

wdf = normalizeToSeason(wdf)
sns.kdeplot(wdf['Score'])
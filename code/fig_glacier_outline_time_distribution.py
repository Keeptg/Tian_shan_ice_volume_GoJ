#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 01:18:34 2020

@author: keeptg
"""

import os
import numpy as np
import geopandas as gpd

%matplotlib widget
import matplotlib.pyplot as plt
from path_config import *


fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.hist(yyyy, range(np.min(yyyy), np.max(yyyy)), normed=1, label='1')
ax2.hist(yyyy_r, range(np.min(yyyy_r), np.max(yyyy_r)), normed=1, label='2')
ax1.set_xlim(np.min(yyyy)-1, np.max(yyyy)+1)
ax1.set_xticks(np.arange(np.min(yyyy)+1, np.max(yyyy)+1, 3))
ax2.set_xlim(np.min(yyyy_r)-1, np.max(yyyy_r)+1)
ax2.set_xticks(np.arange(np.min(yyyy_r)+1, np.max(yyyy_r)+1, 3))
ax1.text(0.05, 0.95, f'N={len(yyyy)}', transform=ax1.transAxes, fontsize=10,
         verticalalignment='top')
ax2.text(0.05, 0.95, f'N={len(yyyy_r)}', transform=ax2.transAxes, fontsize=10,
         verticalalignment='top')
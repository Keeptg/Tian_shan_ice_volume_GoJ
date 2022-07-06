#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:34:42 2020

@author: keeptg
"""

import os, sys
import geopandas as gpd

from rgitools.funcs import compute_intersects, check_geometries
sys.path.append('/home/keeptg/Data/Study_in_Innsbruck/Tienshan/Script')
from Re_GlabTop import glc_inventory2oggm

root_dir = '/home/keeptg/Data/Study_in_Innsbruck/'
data_dir = os.path.join(root_dir, 'Data', 'Shpfile', 'Tienshan_data')

path = os.path.join(data_dir, 'gamdam')
df = gpd.read_file(path)
df = glc_inventory2oggm(df)
cdf = check_geometries(df)
intersect = compute_intersects(cdf)
intersect.to_file(os.path.join(data_dir, 'gamdam_intersects'))

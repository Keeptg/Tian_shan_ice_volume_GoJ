#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 18:50:18 2020

@author: keeptg
"""


import os, sys, pickle
import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd

script_dir = os.path.split(sys.path[0])[0]
if script_dir not in sys.path:
    sys.path.insert(1, script_dir)
from path_config import *
import matplotlib.pyplot as plt

data_dir = os.path.join(root_dir, 'Tienshan_data')
data_ = os.path.join(root_dir, 'cluster_output')

ggidf = gpd.read_file(os.path.join(data_dir, 'gamdam'))
rgidf = gpd.read_file(os.path.join(data_dir, 'rgi'))
ggidf['num'] = 1
rgidf['num'] = 1

try:
    with open(os.path.join(data_, 'in_china_list.pkl'), 'rb') as f:
        incn = pickle.load(f)
except FileNotFoundError:
    china_gdf = gpd.read_file(os.path.join(root_dir, 'Data', 'Shpfile',
                                           'china-shapefiles-master', 'china_country.shp'))

    ggi_centroid = ggidf.centroid.values
    ggi_in_china = np.array([china_gdf.contains(geom).values[0] for geom in ggi_centroid])

    rgi_centroid = rgidf.centroid.values
    rgi_in_china = np.array([china_gdf.contains(geom).values[0] for geom in rgi_centroid])

    incn = {'ggi_in_China': ggi_in_china, 'rgi_in_China': rgi_in_china}

    with open(os.path.join(data_, 'in_china_list.pkl'), 'wb') as f:
        pickle.dump(incn, f)

ggidf['in_china'] = incn['ggi_in_China']
rgidf['in_china'] = incn['rgi_in_China']
rgidf['yyyy'] = rgidf.BgnDate.str[:4]
rgidf['mm'] = rgidf.BgnDate.str[4:6].astype(np.int)
ggidf['yyyy'] = ggidf.yyyy.astype(str)

ggicn = ggidf[ggidf.in_china]
ggifr = ggidf[~ggidf.in_china]
rgicn = rgidf[rgidf.in_china]
rgifr = rgidf[~rgidf.in_china]

tn_ggicn = ggicn.groupby(by='yyyy').sum()
tn_ggifr = ggifr.groupby(by='yyyy').sum()
tn_rgicn = rgicn.groupby(by='yyyy').sum()
tn_rgifr = rgifr.groupby(by='yyyy').sum()

years = [str(year) for year in range(1994, 2009, 1)]
colors = ['salmon', 'lightskyblue']
fsize=7
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
ax = axs[0]
ax.tick_params(axis='both', which='major', labelleft=False, right='True', left='False', 
               labelbottom=False, labelright=False)
ax.yaxis.tick_right()
ax.invert_xaxis(), ax.set_yticklabels([]), ax.set_xlim(5500, 0)
ax.spines['right'].set_linewidth(1.5)
[ax.spines[axis].set_visible(False) for axis in ['top', 'left', 'bottom']]
ax.set_xticks([])
for year in years:
    nums = []
    num = 0
    for df in [tn_rgicn, tn_rgifr]:
        try:
            num = df.loc[year].num
        except:
            num = 0
        nums.append(num)
    ax.barh(year, nums[0], color=colors[0])
    ax.barh(year, nums[1], left=nums[0], color=colors[1])
    if nums[0] + nums[1] == 0:
        continue
    ax.text(nums[0]+nums[1]+50, year, f'{int(nums[1])}/{int(nums[0])}', ha='right', va='center', 
            fontsize=fsize, fontweight='bold', fontstyle='italic')
ax.text(50, 15.5, 'A', fontweight='bold', fontsize=12, ha='right', va='top')
ax.bar(0, 0, color=colors[0], label='In China')
ax.bar(0, 0, color=colors[1], label='Outside China')
ax.legend(loc='lower right')

ax = axs[1]
ax.set_xlim(0, 5500)
ax.set_xticks([])
[ax.spines[axis].set_visible(False) for axis in ['top', 'right', 'bottom']]
ax.spines['left'].set_linewidth(1.5)
for year in years:
    nums = []
    num = 0
    for df, color in zip([tn_ggicn, tn_ggifr], ['salmon', 'lightskyblue']):
        try:
            num = df.loc[year].num
        except:
            num = 0
        nums.append(num)
    ax.barh(year, nums[0], color=colors[0])
    ax.barh(year, nums[1], left=nums[0], color=colors[1])
    if nums[0] + nums[1] == 0:
        continue
    ax.text(nums[0]+nums[1]+50, year, f'{int(nums[0])}/{int(nums[1])}', ha='left', va='center',
            fontsize=fsize, fontweight='bold', fontstyle='italic')
ax.text(50, 15.5, 'B', fontweight='bold', fontsize=12, ha='left', va='top')
fig.savefig(os.path.join(data_dir, 'figure', 'glc_inv_date_distrib.pdf'), dpi=300, bbox_inches='tight')


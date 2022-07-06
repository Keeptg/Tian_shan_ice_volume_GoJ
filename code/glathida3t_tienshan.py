#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 00:11:44 2020

@author: keeptg
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

root_path = '/home/keeptg/Data/Study_in_Innsbruck'
data_dir = os.path.join(root_path, 'Data', 'Shpfile', 'Tienshan_data')
tienshan_gdf = gpd.read_file(os.path.join(root_path, 'Shpfile', 
                                          'tien_shan.shp'))
detected_glaciers = gpd.read_file(os.path.join(root_path, 'Shpfile',
                                               'glacier_rgi_Tienshan_GTD.shp'))

glathida_df = pd.read_csv(os.path.join(root_path, 'GlaThiDa3.01', 'data', 
                                       'TTT.csv'))

in_tienshan = [tienshan_gdf.geometry.contains(Point(x, y))[0]
               for x, y in zip(glathida_df.POINT_LON,
                               glathida_df.POINT_LAT)]

glathida_tienshan = glathida_df.loc[in_tienshan]
glathida_tienshan.to_csv(os.path.join(root_path, 'GlaThiDa3.01', 
                                      'glathida_tienshan.csv'))

del glathida_df, in_tienshan

ax = tienshan_gdf.plot()
detected_glaciers.plot(facecolor='none', edgecolor='black', ax=ax)
ax.scatter(glathida_tienshan.POINT_LON, glathida_tienshan.POINT_LAT, c='r')

path = os.path.join(root_path, 'Tienshan', 'Manual_glaciers_outline_info')
f_lists = os.listdir(path)
list_ = [l for l in f_lists if (('.shp' in l) and ('_2000.shp' not in l))]

for l in list_:
    gdf = gpd.read_file(os.path.join(path, l))
    gdf.to_crs({'init': 'epsg:4326'}, inplace=True)
    try:
        manual_glaciers = pd.concat([manual_glaciers, gdf])
    except NameError:
        manual_glaciers = gdf

path = os.path.join(root_path, 'Shpfile', 'GAMDAM', 'Central_Asia')
gamdam_gdf = gpd.read_file(path)
in_tienshan = [tienshan_gdf.geometry.contains(point)[0] for point in 
               gamdam_gdf.geometry.centroid]
gamdam_tienshan = gamdam_gdf.loc[in_tienshan]
del gamdam_gdf, in_tienshan

glathida_tienshan = pd.read_csv(os.path.join(root_path, 'GlaThiDa3.01', 
                                             'glathida_tienshan.csv'))
point_gdf = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y 
                                       in zip(glathida_tienshan.POINT_LON,
                                              glathida_tienshan.POINT_LAT)],
                             crs={'init': 'epsg:4326'})
point_gdf['thick'] = glathida_tienshan.THICKNESS
point_gdf['date'] = glathida_tienshan.SURVEY_DATE.astype(str)

# point_gdf.to_file(os.path.join(root_path, 'Shpfile', 'Tienshan_data',
#                                'glathida'))

gamdam_glathida_tienshan = gpd.sjoin(gamdam_tienshan, point_gdf)
gamdam_glathida_tienshan.drop_duplicates('geometry', inplace=True)

for i in range(len(gamdam_glathida_tienshan)):
    gamdam_glathida_tienshan.iloc[[i]].plot()

# gamdam_glathida_tienshan.to_file(os.path.join(root_path, 'Shpfile', 
#                                               'Tienshan_data',
#                                               'gamdam_glathida', 
#                                               'gamdam_glathida.shp'))

# gamdam_tienshan.to_file(os.path.join(root_path, 'Shpfile', 'Tienshan_data',
#                                      'gamdam'))

# manual_glaciers.to_file(os.path.join(root_path, 'Shpfile', 'Tienshan_data',
#                                      'mgi2013_glathida'))

from oggm import cfg, utils, workflow
cfg.initialize()
rgidf = gpd.read_file(utils.get_rgi_region_file(version='61', region='13'))
in_tienshan = [tienshan_gdf.geometry.contains(point)[0] for point in 
               rgidf.geometry.centroid]
rgidf_tienshan = rgidf.loc[in_tienshan]
# rgidf_tienshan.to_file(os.path.join(root_path, 'Data', 'Shpfile', 
#                                     'Tienshan_data', 'rgi'))

gamdam = gpd.read_file(os.path.join(data_dir, 'gamdam_glathida'))
gtd = gpd.read_file(os.path.join(data_dir, 'glathida'))
gtd['glc_name'] = ''
glc_name = []
for index in gtd.index:
    pnt = gtd.loc[index].geometry
    in_glc = gamdam.geometry.contains(pnt)
    glc = gamdam[in_glc]
    if len(glc) > 0:
        glc_name.append(glc.name.values[0])
    else:
        glc_name.append(np.nan)

gtd['glc_name'] = glc_name

for name in gamdam.name.values:
    sub_gtd = gtd[gtd.glc_name==name]
    num = len(sub_gtd)
    mean_thick = sub_gtd.thick.mean()
    date = list(set(sub_gtd.date))
    print('glc_name: {}, point_num: {}, mean_thick: {:.2f}, date: {}'.format(name, num, mean_thick, date))

def print_info(name, year):
    sub_gtd = gtd[gtd.glc_name==name]
    sub_gtd14 = sub_gtd[sub_gtd.date.str.contains(year)]
    num = len(sub_gtd14)
    mean_thick = sub_gtd14.thick.mean()
    print('glc_name: {}, year: {}, num: {}, mean_thick: {}'.format(name, year, num, mean_thick))

print_info('UME', '2014')
print_info('UME', '2006')
print_info('UMW', '2014')
print_info('UMW', '2006')






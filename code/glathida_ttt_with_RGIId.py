#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:22:06 2020

@author: keeptg
"""

import os, copy
import numpy as np
import pandas as pd

from shapely.geometry import Point
import geopandas as gpd
from oggm import utils

root_dir = '/home/keeptg/Data/Study_in_Innsbruck/'
data_dir = os.path.join(root_dir, 'Data', 'Shpfile', 'Tienshan_data')
root_working_dir = os.path.join(root_dir, 'Tienshan', 'work_dir')


glathida_info = gpd.read_file(utils.get_glathida_file())
glathida13_info = glathida_info[glathida_info.RGI60_ID.str.contains('60-13.')]
glathida14_info = glathida_info[glathida_info.RGI60_ID.str.contains('60-14.')]
glathida15_info = glathida_info[glathida_info.RGI60_ID.str.contains('60-15.')]

glathida_data = pd.read_csv(os.path.join(root_dir, 'Data', 'GlaThiDa3.01', 
                                         'data', 'TTT.csv'))
glathida_h = glathida_data[glathida_data.ELEVATION>3500]
glathida_id = list(set(glathida_data.GlaThiDa_ID))
rgi_region = gpd.read_file('/home/keeptg/OGGM/rgi/RGIV60/00_rgi60_regions/00_rgi60_O1Regions.shp')
t = pd.read_csv(os.path.join(root_dir, 'Data', 'GlaThiDa3.01', 
                             'data', 'T.csv'))
tc = t[t.GlaThiDa_ID.isin(glathida_id)]

geom = [Point(x, y) for x, y in zip(glathida_data.POINT_LON, 
                                    glathida_data.POINT_LAT)]

glathida_gdf = gpd.GeoDataFrame(glathida_data, geometry=geom, crs='epsg:4326')

region_gtd = gpd.sjoin(glathida_gdf, rgi_region, how='left', op='within')
region_gtd.to_file(os.path.join(root_dir, 'Data', 'GlaThiDa3.01', 
                                         'data', 'TTT_with_region.shp'))
region_gtd.drop(['index_right', 'FULL_NAME'], inplace=True, axis=1)
code_list = ['{:0>2d}'.format(code) for code in region_gtd.RGI_CODE.values]
region_gtd['RGI_CODE'] = code_list
# region_gtd.to_file(os.path.join(root_dir, 'Data', 'GlaThiDa3.01', 
#                                          'data', 'TTT_with_region.shp'))
cp = copy.copy(region_gtd).drop(['geometry'], axis=1)
cp.to_csv(os.path.join(root_dir, 'Data', 'GlaThiDa3.01', 
                                         'data', 'TTT_with_region.csv'))

code_list = list(set(code_list))


region_gtd = pd.read_csv(os.path.join(root_dir, 'Data', 'GlaThiDa3.01',
                                        'data', 'TTT_with_region.csv'))
geom = [Point(x, y) for x, y in zip(region_gtd.POINT_LON, region_gtd.POINT_LAT)]
region_gtd = gpd.GeoDataFrame(region_gtd, geometry=geom, crs='epsg:4326')
code_list = list(set(region_gtd.RGI_CODE.values))



gtd_sub_lists = []
for region in code_list:
    region = int(region)
    gtd_sub = region_gtd[region_gtd.RGI_CODE==region]
    region = '{:0>2d}'.format(region)
    path = utils.get_rgi_region_file(region=region, version='6')
    rgidf = gpd.read_file(path)
    rgidf.drop(['GLIMSId', 'BgnDate', 'EndDate', 'CenLon', 'CenLat',
       'O1Region', 'O2Region', 'Area', 'Zmin', 'Zmax', 'Zmed', 'Slope',
       'Aspect', 'Lmax', 'Status', 'Connect', 'Form', 'TermType', 'Surging',
       'Linkages', 'Name'], axis=1, inplace=True)
    gtd_rgiid = gpd.sjoin(gtd_sub, rgidf, how='left', op='within')
    gtd_rgiid = gtd_rgiid[~gtd_rgiid.RGIId.isna().values]
    gtd_sub_lists.append(gtd_rgiid)
    print(region)

concat = pd.concat(gtd_sub_lists)
concat.drop(['geometry'], axis=1, inplace=True)
    
concat = pd.concat(gtd_sub_lists)
concat = pd.DataFrame(concat)
concat.drop(['Unnamed: 0', 'index_right'], axis=1, inplace=True)
concat.to_csv(os.path.join(root_dir, 'Data', 'GlaThiDa3.01',
                                        'data', 'TTT_with_RGIId.csv'))

e3000 = concat[concat.ELEVATION>3000]
rgilist = list(set(e3000.RGIId.values))
gtd_e3000 = concat[concat.RGIId.isin(rgilist)]
geom = [Point(x, y) for x, y in zip(gtd_e3000.POINT_LON, gtd_e3000.POINT_LAT)]
gtd_3000_gdf = gpd.GeoDataFrame(gtd_e3000, geometry=geom, crs='epsg:4326')
path = os.path.join(root_dir, 'OGGM_calibrating_test', 'gtd_e3000')
if not os.path.exists(path):
    os.mkdir(path)
gtd_3000_gdf.to_file(path)


import pickle
f = open(os.path.join(root_dir, 'OGGM_calibrating_test', 'gtd_e3000_rgiid.pkl'),
         'wb')
pickle.dump(rgilist, f)
f.close()
f = open(os.path.join(root_dir, 'OGGM_calibrating_test', 'gtd_e3000_rgiid.pkl'),
         'rb')

a = pickle.load(f)
f.close()

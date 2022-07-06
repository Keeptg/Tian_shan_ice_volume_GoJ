#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 01:42:49 2020

@author: keeptg
"""

import os
import numpy as np
import pandas as pd
from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer
from path_config import root_dir

username = 'lifei15@outlook.com'
password = 'duomi1duomi2'
execute_download_progress = False

# Supportted landsat: LANDSAT_TM_C1, LANDSAT_ETM_C1, LANDSAT_8_C1
landsat_types = ['LANDSAT_TM_C', 'LANDSAT_ETM_C1', 'LANDSAT_8_C1']
download_years = ['2000', '1999', '2001', '2013', '2014', '2015']

root_path = os.path.join(root_dir, 'Tienshan')
path = os.path.join(root_path, 'glacier_loc.csv')
locations = pd.read_csv(path)

lats = locations.lat.values
lons = locations.lon.values

y2000 = []
y2013 = []

info_df = pd.DataFrame(columns=['WRS2 path-row', 'Data', 'Scene ID', 'Sensor', 'Glacier'])

for name, lat, lon in zip(locations.Glc_name.values, lats, lons):
    for year in download_years:
        if year in ['2013', '2014', '2015']:
            landsat_type = landsat_types[2]
            path_collector = y2000
        elif year in ['1998', '1999', '2000', '2001', '2002']:
            landsat_type = landsat_types[1]
            sensor = 'ETM+'
        
        api = API(username, password)
        scenes = api.search(dataset=landsat_type,
                            latitude=lat,
                            longitude=lon,
                            start_date=f'{year}-04-01',
                            end_date=f'{year}-08-31',
                            max_cloud_cover=20)
        ls = []
        for info in scenes:
            path = int(info['summary'].split(', ')[-2].split(': ')[-1])
            row = int(info['summary'].split(', ')[-1].split(': ')[-1])
            path_row = '{:0>3d}-{:0>3d}'.format(path, row)
            data = info['startTime']
            id = info['entityId']
            if id[1] == 'C':
                sensor = 'OLI/TIRS'
            elif id[1] == 'O':
                sensor = 'OLI'
            elif id[1] == 'T':
                if landsat_type == landsat_types[2]:
                    sensor = 'TIRS'
                elif landsat_type == landsat_types[0]:
                    sensor = 'TM'
                else:
                    raise NameError(f"Unknown sensor type for '{id}' in '{landsat_type}'!")
            elif id[1] == 'E':
                sensor = 'ETM+'
            elif id[1] == 'M':
                sensor = 'MSS'
            else:
                raise NameError(f"Unknown sensor type for '{id}' in '{landsat_type}'!")
            glacier = name
            index = len(info_df)
            info_df.loc[index] = [path_row, data, id, sensor, glacier]
        api.logout()
        if execute_download_progress:
            outpath = os.path.join(root_path, 'Landsat8',
                                   name.replace(' ', '_')+str(year))
            if (not os.path.exists(outpath)) and (len(scenes) > 0):
                os.mkdir(outpath)
            else:
                continue
            ee = EarthExplorer(username, password)
            for scene in scenes:
                scene_id = scene['entityId']
                ee.download(scene_id, output_dir=outpath)

            ee.logout()

pd.set_option('display.max_columns', 500)
st2000 = info_df[np.logical_and(info_df.Glacier=='Sary Tor',
                         info_df.Data.str.contains('2001'))]
st2013 = info_df[np.logical_and(info_df.Glacier=='Sary Tor',
                                info_df.Data.str.contains('2013'))]
tyk2000 = info_df[np.logical_and(info_df.Glacier=='Tsentralniy Tuyuksu',
                         info_df.Data.str.contains('1999'))]
tyk2013 = info_df[np.logical_and(info_df.Glacier=='Tsentralniy Tuyuksu',
                                 info_df.Data.str.contains('2013'))]
hg2000 = info_df[np.logical_and(info_df.Glacier=='Heigou No. 8',
                                 info_df.Data.str.contains('1999'))]
hg2013 = info_df[np.logical_and(info_df.Glacier=='Heigou No. 8',
                                info_df.Data.str.contains('2013'))]
hxlg2000 = info_df[np.logical_and(info_df.Glacier=='Haxilegeng No. 52',
                                info_df.Data.str.contains('2000'))]
hxlg2013 = info_df[np.logical_and(info_df.Glacier=='Haxilegeng No. 52',
                                  info_df.Data.str.contains('2013'))]
hxlg2015 = info_df[np.logical_and(info_df.Glacier=='Haxilegeng No. 52',
                                  info_df.Data.str.contains('2015'))]
sgh2000 = info_df[np.logical_and(info_df.Glacier=='Sigonghe No. 4',
                                info_df.Data.str.contains('2001'))]
sgh2013 = info_df[np.logical_and(info_df.Glacier=='Sigonghe No. 4',
                                 info_df.Data.str.contains('2013'))]
wmw2000 = info_df[np.logical_and(info_df.Glacier=='Urumuqi No. 1 East',
                                 info_df.Data.str.contains('2001'))]
wmw2013 = info_df[np.logical_and(info_df.Glacier=='Urumuqi No. 1 East',
                                 info_df.Data.str.contains('2013'))]
qbt2013 = info_df[np.logical_and(info_df.Glacier=='Qingbingtan No. 72',
                                 info_df.Data.str.contains('2013'))]
qbt2015 = info_df[np.logical_and(info_df.Glacier=='Qingbingtan No. 72',
                                 info_df.Data.str.contains('2015'))]

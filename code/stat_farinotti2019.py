#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 20:56:29 2020

@author: keeptg
"""


import os, glob
import numpy as np
import pandas as pd
import xarray as xr
from rasterio import RasterioIOError

fn_path = '/home/keeptg/Data/Farinotti_2019'
root_dir = '/home/keeptg/Data/Study_in_Innsbruck/'
data_dir = os.path.join(root_dir, 'Data', 'cluster_output',
                        'tienshan_ice_volume')
path_list = glob.glob(os.path.join(fn_path, '*'))
path_list = [path for path in path_list if os.path.isdir(path)]

rgidf = pd.read_csv(glob.glob(os.path.join(data_dir, f'*oggm_rgi.csv'))[0])
rgiid = rgidf.rgi_id.values

fn_rs = pd.DataFrame(index=rgiid)
for path in path_list:
    model_type = os.path.split(path)[1]
    vol_s = []
    for glc in rgiid:
        if 'results_model' in model_type:
            file = 'thickness_' + glc + '.tif'
        else:
            file = glc + '_thickness' + '.tif'
        fpath = os.path.join(path, 'RGI60-13', file)
        try:
            ds = xr.open_rasterio(fpath)
            area = ds.res[0] * ds.res[1] * 1e-6
            h = ds.data.sum() * 1e-3
            vol = area * h
        except RasterioIOError:
            vol = np.nan
        vol_s.append(vol)
    fn_rs[model_type] = vol_s
fn_rs.to_csv(os.path.join(fn_path, 'tienshan_ice_vol.csv'))
    
    
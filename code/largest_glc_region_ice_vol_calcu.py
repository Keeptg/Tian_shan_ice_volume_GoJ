#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 23:46:05 2020

@author: keeptg
"""


import os, sys, pickle
from shutil import copyfile
import pandas as pd
from shapely.geometry import Polygon, Point
import geopandas as gpd
from oggm import cfg, utils, tasks, workflow
from oggm.workflow import execute_entity_task

# from rgitools.funcs import check_geometries
from Re_GlabTop import prepare_Glabtop_inputs
from tienshan_glacier_thick import run_model
from path_config import *

def main():
    dir_ = os.path.join(root_dir, 'Data', 'cluster_output')
    work_dir = os.path.join(work_dir, 'largest_glc_region')
        
    # rgi run
    oggm_rgi1 = pd.read_csv(os.path.join(dir_, 'tienshan_ice_volume',
                                         'Tienshan_glc_volume_uncertainty',
                                         'oggm_rgi_SRTM_opt(5.30).csv'))
    rgiid100 = oggm_rgi1[oggm_rgi1.rgi_area_km2>100].rgi_id.values
    cdf = gpd.read_file(os.path.join(data_dir, 'rgi'))
    region_glc = cdf.loc[[rgiid in rgiid100 for rgiid in cdf.RGIId.values]]
    maxx = region_glc.geometry.bounds.maxx.max()
    maxy = region_glc.geometry.bounds.maxy.max()
    minx = region_glc.geometry.bounds.minx.min()
    miny = region_glc.geometry.bounds.miny.min()
    geom = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny),
                    (minx, miny)])
    region_df = gpd.GeoDataFrame(geometry=[geom], crs='epsg:4326')
    outpath = os.path.join(work_dir, 'largest_glacier_region_shp')
    utils.mkdir(outpath)
    region_df.to_file(os.path.join(outpath, 'region_df.shp'))
    
    rgidf = gpd.read_file(os.path.join(data_dir, 'rgi'))
    ggidf = gpd.read_file(os.path.join(data_dir, 'checked_gamdam'))
    ggi_intersects_dir = os.path.join(data_dir, 'gamdam_intersects')
    rgi_intersects_dir = utils.get_rgi_intersects_region_file(region='13',
                                                              version='61')
    rgidf = rgidf.loc[[region_df.geometry.contains(Point(x, y)).values[0] for 
                              x, y in zip(rgidf.CenLon, rgidf.CenLat)]]
    ggidf = ggidf.loc[[region_df.geometry.contains(Point(x, y)).values[0] for
                              x, y in zip(ggidf.CenLon, ggidf.CenLat)]]
    
    oggm_rgi_gdirs = run_model(glc_inv_gdf=rgidf, model_type='oggm',
                               working_dir=os.path.join(work_dir, 'oggm_rgi'),
                               intersects_dir=rgi_intersects_dir)
    oggm_ggi_gdirs = run_model(glc_inv_gdf=ggidf, model_type='oggm',
                               working_dir=os.path.join(work_dir, 'oggm_ggi'),
                               intersects_dir=ggi_intersects_dir)
    glab_rgi_gdirs = run_model(glc_inv_gdf=rgidf, model_type='glabtop',
                               working_dir=os.path.join(work_dir, 'glab_rgi'),
                               intersects_dir=rgi_intersects_dir)
    glab_ggi_gdirs = run_model(glc_inv_gdf=ggidf, model_type='glabtop',
                               working_dir=os.path.join(work_dir, 'glab_ggi'),
                               intersects_dir=ggi_intersects_dir)
    
    output_dic = dict(zip(['oggm_rgi_gdirs', 'oggm_ggi_gdirs', 
                           'glab_rgi_gdirs', 'glab_ggi_gdirs'], 
                          [oggm_rgi_gdirs, oggm_ggi_gdirs, 
                           glab_rgi_gdirs, glab_ggi_gdirs]))
    output_path = os.path.join(work_dir, 'largest_glc_vol_calcu_gdirs.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(output_dic, f)
        
    return output_path

if __name__ == '__main__':
    main()
        
    
    
    
    
    

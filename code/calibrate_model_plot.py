#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 18:40:30 2020

boxplot data from "calibrate_glabtop.py"

@author: keeptg
"""

import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import geopandas as gpd
from oggm import cfg, utils, tasks, workflow
import salem
import xarray as xr
from shapely.geometry import Polygon, Point
sys.path.append('/home/keeptg/Data/Study_in_Innsbruck/Tienshan/Script')
from Re_GlabTop import glc_inventory2oggm
from Re_GlabTop import gridded_measured_thickness_point as gridded_point


def geo_thick_df(thick_df):
    geom = [Point(x, y) for x, y in zip(thick_df.cenlon, thick_df.cenlat)]
    thick_gdf = gpd.GeoDataFrame(thick_df, geometry=geom, crs='epsg:4326')
    
    return thick_gdf


def status_vol_area_info(elon, elat, thick_df, inter=1):
    data_gdf = gpd.GeoDataFrame(columns=['areas', 'volumes', 'numb', 
                                         'geometry'],
                                crs='epsg:4326')
    t_gdf = geo_thick_df(thick_df)
    p_inter = inter
    for x in np.arange(*elon, p_inter):
        for y in np.arange(*elat, p_inter):
            polygon = Polygon([(x-p_inter/2, y-p_inter/2), 
                               (x-p_inter/2, y+p_inter/2), 
                               (x+p_inter/2, y+p_inter/2), 
                               (x+p_inter/2, y-p_inter/2), 
                               (x-p_inter/2, y-p_inter/2)])
            point = Point(x, y)
            in_polygon = [polygon.contains(p) for p in t_gdf.geometry]
            sub_gdf = t_gdf[in_polygon].copy()
            t_gdf = t_gdf[~np.array(in_polygon)]
            numb = len(sub_gdf)
            if numb>0:
                areas = sub_gdf.rgi_area_km2.sum()
                volumes = sub_gdf.inv_volume_km3.sum()
                data_gdf.loc[len(data_gdf)] = [areas, volumes, numb, point]
            if len(t_gdf) == 0:
                break
        if len(t_gdf) == 0:
            break
        
    return data_gdf


def get_farinotti_data_path(rgiid, model_type, region='13', dpath=None):
    
    name_dic = {'CM': 'Composite', 'M1': 'results_model1', 
                'M2': 'results_model2', 'M3': 'results_model3', 
                'M4': 'results_model4', 'M5': 'results_model5'}
    
    if not dpath:
        dpath = '/home/keeptg/Data/Farinotti_2019'
    if model_type == 'CM':
        path = os.path.join(dpath, name_dic[model_type], f'RGI60-{region}',
                            f'{rgiid}_thickness.tif')
    else:
        path = os.path.join(dpath, name_dic[model_type], f'RGI60-{region}',
                            f'thickness_{rgiid}.tif')

    return path


def status_farinotti_models_results(rgidf, model_type, 
                                    region='13', dpath=None):
    
    volumes = []
    for rgiid in rgidf.RGIId:
        path = get_farinotti_data_path(rgiid, model_type, region, dpath)
        if not os.path.exists(path):
            volume = np.nan
        else:
            thick_data = salem.open_xr_dataset(path)
            dx = abs(thick_data.salem.grid.dx)
            dy = abs(thick_data.salem.grid.dy)
            volume = np.nansum(thick_data.data.values) * dx * dy * 1e-9
        
        volumes.append(volume)
    
    return volumes


def give_farinotti_models_results(path=None, write=True):
    if not path:
        path = os.path.join(data_dir, 'tienshan_glacier_thick_gamdam_gi',
                            'rgidf_with_farinotti2019')
    if not os.path.exists(path):
        rgidf = gpd.read_file(os.path.join(data_dir, 'rgi'))
        for model_type in ['CM', 'M1', 'M2', 'M3', 'M4']:
            rgidf[f'{model_type}_volumes'] = \
                status_farinotti_models_results(rgidf, model_type)
                
        if write:
            rgidf.to_file(os.path.join(data_dir, 
                                       'tienshan_glacier_thick_gamdam_gi',
                                       'rgidf_with_farinotti2019'))
    else:
        rgidf = gpd.read_file(path)
    
    return rgidf


# The 
def merge_gdirs_thick_data(ob_grid, gdirs):
    
    for gdir in gdirs:
        if not os.path.exists(gdir.get_filepath('gridded_data')):
            continue
        gridded_data = xr.open_dataset(gdir.get_filepath('gridded_data'))
        var_name = 'distributed_thickness'
        if not (var_name in gridded_data.data_vars):
            continue
        thick_da = gridded_data.distributed_thickness
        thick_da.attrs['pyproj_srs'] = gridded_data.proj_srs
        trans_thick = ob_grid.salem.transform(thick_da)
        trans_thick.values = np.where(np.isnan(trans_thick.values), 0, 
                                      trans_thick.values)
        try:
            trans_thicks.values += trans_thick
        except NameError:
            trans_thicks = trans_thick
        raise('test')
        
    return trans_thicks


root_dir = '/home/keeptg/Data/Study_in_Innsbruck/'
data_dir = os.path.join(root_dir, 'Data', 'Shpfile', 'Tienshan_data')
# Prepare map data
# ------
gamdam = salem.read_shapefile(os.path.join(data_dir, 'gamdam', 'gamdam.shp'))
bounds = gamdam.geometry.bounds
border = 2.5
maxx, minx = bounds.maxx.max() + border, bounds.minx.min() - border
maxy, miny = bounds.maxy.max() + border, bounds.miny.min() - border
bound_df = gpd.GeoDataFrame({'geometry': [Polygon([(minx, miny),
                                                  (minx, maxy),
                                                  (maxx, maxy),
                                                  (maxx, miny),
                                                  (minx, miny)])]},
                            crs='epsg:4326')
cfg.initialize()
working_dir = os.path.join(data_dir, 'tienshan_plot')
cfg.PATHS['working_dir'] = utils.mkdir(working_dir, reset=False)
cfg.PARAMS['use_intersects'] = False
cfg.PARAMS['use_rgi_area'] = False
cfg.PARAMS['border'] = 0
gdf = glc_inventory2oggm(bound_df)
gdir = workflow.init_glacier_regions(gdf)[0]
dem = salem.open_xr_dataset(gdir.get_filepath('dem'))
grid = dem.salem.grid

# Read boxplot data
# ------
data = pd.read_csv(os.path.join(data_dir, 'calibrate_model.csv'))

# Prepare volume scatter plot data
# ------
oggm_df = pd.read_csv(os.path.join(data_dir, 
                                    'tienshan_glacier_thick_gamdam_gi',
                                    'oggm_workdir', 'glacier_statistics.csv'))
glabtop1_df = pd.read_csv(os.path.join(data_dir, 
                                    'tienshan_glacier_thick_gamdam_gi',
                                    'glabtop1_workdir', 
                                    'glacier_statistics.csv'))
glabtop2_df = pd.read_csv(os.path.join(data_dir, 
                                    'tienshan_glacier_thick_gamdam_gi',
                                    'glabtop2_workdir', 
                                    'glacier_statistics.csv'))

# volumed_gamdam = oggm_df.copy()
# volumed_gamdam = volumed_gamdam.rename(columns={'inv_volume_km3':'oggm_volume'})
# volumed_gamdam['glabtop1_volume'] = glabtop1_df.inv_volume_km3
# volumed_gamdam['glabtop2_volume'] = glabtop2_df.inv_volume_km3
# volumed_gamdam = gpd.GeoDataFrame(volumed_gamdam, geometry=gamdam.geometry,
#                                   crs='epsg:4326')
# utils.mkdir(os.path.join(data_dir, 'volumed_gamdam'))
# volumed_gamdam.to_file(os.path.join(data_dir, 'volumed_gamdam', 
#                                     'volumed_gamdam.shp'))

elon = oggm_df.cenlon.min(), oggm_df.cenlon.max()
elat = oggm_df.cenlat.min(), oggm_df.cenlat.max()
thick_oggm = status_vol_area_info(elon, elat, oggm_df)
thick_glabtop1 = status_vol_area_info(elon, elat, glabtop1_df)
thick_glabtop2 = status_vol_area_info(elon, elat, glabtop2_df)

# Axis plot
# ------
fig = plt.figure(figsize=(7, 4))
plt.subplots_adjust(wspace=0.03)
ax3 = fig.add_subplot(3, 4, (1, 8))
ax1 = fig.add_subplot(3, 4, (9, 11))
ax2 = fig.add_subplot(3, 4, 12)

# Boxplot
# ------

sns.boxplot(x='name', y='thick_diff', hue='group', data=data,
            whis=[5, 95], ax=ax1, showfliers=False, 
            palette=['lightcyan', 'turquoise', 'teal'])
sns.boxplot(x='group', y='thick_diff', data=data, whis=[5, 95], ax=ax2, 
            showfliers=False, palette=['lightcyan', 'turquoise', 'teal'])
ax2.set_yticklabels([])
ax2.yaxis.set_ticks_position('none')

ylim = -120, 120
ax1.set_ylim(ylim), ax2.set_ylim(ylim)
ax1.set_yticks(np.arange(-100, 101, 50))
ax1.legend_.remove()
xlim = ax1.get_xlim()
ax1.plot(xlim, (0, 0), lw=1.2, color='darkred')
ax1.set_xlim(xlim)
ax1.set_xticklabels(['HXLG', 'HG', 'QBT', 'ST', 'SGH', 'TYK', 'UMW', 'UMW'])
ax2.set_xticklabels(['BS1', 'BS2', 'MB'])
xlim = ax2.get_xlim()
ax2.plot(xlim, (0, 0), lw=1.2, color='darkred')
ax2.set_xlim(xlim)
ax1.set_ylabel('Absolute Error (m)')
ax1.set_xlabel('Glaciers'), ax2.set_xlabel('Models'), ax2.set_ylabel('')
ax1.tick_params(axis='both', which='major', labelsize=8)
ax2.tick_params(axis='both', which='major', labelsize=8)

# Map plot
# ------
sm = salem.Map(grid, countries=True)
sm.set_topography(gdir.get_filepath('dem'), relief_factor=1.4)
cmap_v = np.full(dem.data.values.shape, 0.5)
cmap_v[0, :], cmap_v[:, 0] = 1, 0
sm.set_data(cmap_v)
sm.set_cmap('Greys')
sm.visualize(ax=ax3, addcbar=False)
cv, sv = thick_oggm.volumes.values, np.sqrt(thick_oggm.areas.values)*10
x, y = sm.grid.transform(thick_oggm.geometry.x, thick_oggm.geometry.y)
ax3.scatter(x, y, c=cv, s=sv, cmap='plasma', norm=LogNorm(), edgecolor='k',
            lw=1, alpha=.8)

# Barplot
# ------
def split_df(df, area_space, by=None):
    
    if not by:
        by = 'rgi_area_km2'
    outlist = {}
    for i in range(len(area_space) - 1): 
        outlist[str(area_space[i])] = \
            df[np.logical_and(df[by]>area_space[i],
                              df[by]<area_space[i+1])]
    return outlist

area_space = [0, 1, 5, 10, 100, 500]
oggm_splited = split_df(oggm_df, area_space)
glabtop1_splited = split_df(glabtop1_df, area_space)
glabtop2_splited = split_df(glabtop2_df, area_space)
volumed_rgidf = give_farinotti_models_results()
# volumed_rgidf.to_file(os.path.join(data_dir, 'volumed_gi', 'volumed_rgi.shp'))
f_splited = split_df(volumed_rgidf, area_space, 'Area')

fig = plt.figure(figsize=(6, 4))
ax1 = fig.add_subplot(111)
datalist = [oggm_splited, glabtop1_splited, glabtop2_splited]
labels = ['MB', 'BS1', 'BS2']
h_list = [.5, .25, .12]
c_list = ['lightcyan', 'turquoise', 'teal']
labels_ = ['M1', 'M2', 'M3', 'M4']
h_list_ = h_list+[.036]
c_list_ = ['mistyrose', 'lightcoral', 'r', 'maroon']
columns_ = ['M1_volumes', 'M2_volumes', 'M3_volumes', 'M4_volumes']
keys_list = list(f_splited.keys())

plot_loc = np.array([0, 1.2, 2.4, 3.6, 4.8])
bar_width = h_list[0]
bar_loc1 = plot_loc + bar_width/2
for i in range(len(datalist)):
    df = datalist[i]
    ax1.barh(bar_loc1, 
             [d.inv_volume_km3.sum() for d in df.values()], 
             height=h_list[i], color=c_list[i], 
             label=labels[i])

for j, i in enumerate(keys_list):
    df_ = f_splited[i]
    bar_loc2 = plot_loc[j] - bar_width/2
    values = df_[columns_].sum().values

    ax1.barh(bar_loc2, values[0], 
             height=h_list_[0], label=labels_[0], color=c_list_[0])
    ax1.barh(bar_loc2, values[1], 
             height=h_list_[1], label=labels_[1], color=c_list_[1])
    ax1.barh(bar_loc2, values[2], 
             height=h_list_[2], label=labels_[2], color=c_list_[2])
    ax1.barh(bar_loc2, values[3], 
             height=h_list_[3], label=labels_[3], color=c_list_[3])

ax1.set_yticks(plot_loc)
ax1.set_yticklabels(['0~1', '1~5', '5~10', '10~100', '100~500'], rotation=90)
ax11 = ax1.twiny()
ax11.barh(bar_loc1, [len(d) for d in df.values()], 
         height=h_list[0], color='dimgrey', label='GAMDAM')
ax11.barh(plot_loc - bar_width/2, [len(f_splited[d]) for d in keys_list], 
         height=h_list_[0], color='lightgrey', label='RGI6.0')

ax11.set_xscale('log')
ax11.set_zorder(0)
ax11.set_xlim(4, 1e6)
ax11.set_xticks([1e1, 1e2, 1e3, 1e4])
ax11.yaxis.tick_right()
ax11.set_yticklabels(['0~1', '1~5', '5~10', '10~100', '100~500'], rotation=-90)
ax11.invert_xaxis()
ax11.spines['top'].set_color('dimgrey')
ax11.spines['right'].set_color('dimgrey')
ax1.spines['top'].set_color('dimgrey')
ax11.tick_params(axis='x', colors='dimgrey')
ax11.minorticks_off()
ax11.set_xlabel('Clacier Numbers in Each range', color='dimgrey')
ax1.tick_params(axis='x', colors='teal')
ax1.spines['bottom'].set_color('teal')
ax11.spines['bottom'].set_color('teal')
ax1.set_ylabel('Glacier Area Ranges ($\mathregular{km^2}$)')
ax1.set_xlabel('Calculated Glacier Volumes ($\mathregular{km^3}$)', 
               color='teal')
# ax1.legend(labels+labels_, loc=[.325, .26], ncol=4)
ax1.legend(labels+labels_, loc=[-.0, 1.], ncol=4, 
           columnspacing=.5, handletextpad=.1,
           handlelength=1, fontsize=9)
handles, labels = ax1.get_legend_handles_labels()
ax11.legend(loc=[.775, -.17], framealpha=1)
plt.tight_layout()

from salem import sample_data_dir
shp = os.path.join(sample_data_dir, 'shapes', 'world_borders', 'world_borders.shp')

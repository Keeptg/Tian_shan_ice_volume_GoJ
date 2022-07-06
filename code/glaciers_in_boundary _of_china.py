#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:03:20 2020

@author: keeptg
"""

import os
import pickle
import sys
from copy import deepcopy as copy
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import geopandas as gpd
import xarray as xr
import salem

from oggm import utils, cfg, workflow, tasks
from oggm.workflow import execute_entity_task

import matplotlib.pyplot as plt
from Re_GlabTop import glc_inventory2oggm, prepare_Glabtop_inputs
from Re_GlabTop import base_stress_inversion, get_farinotti_file_path
from path_config import *


def regional_thick_arr(regional_dem_path, gdirs):
    template = salem.open_xr_dataset(regional_dem_path)
    shape = template.data.values.shape
    thicks = np.zeros(shape)
    masks = np.zeros(shape)
    for i, gdir in enumerate(gdirs):
        print(len(gdirs), i)
        gridded_data = xr.open_dataset(gdir.get_filepath('gridded_data'))
        proj_srs = gridded_data.pyproj_srs
        try:
            thick = gridded_data.distributed_thickness
            mask = gridded_data.glacier_mask
        except AttributeError:
            continue
        thick.attrs['pyproj_srs'] = proj_srs
        mask.attrs['pyproj_srs'] = proj_srs
        thick_trans = template.salem.transform(thick)
        mask_trans = template.salem.transform(mask)
        thicks += np.where(np.isnan(thick_trans.values), 0, 
                           thick_trans.values)
        masks += np.where(np.isnan(mask_trans.values), 0, mask_trans.values)
        
    return thicks, masks


def run_inversion_model_with_calib_param(model_type, p_value, gidf, gitype):

    cfg.initialize()
    cfg.PARAMS['use_rgi_area'] = False
    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['dl_verify'] = False
    cfg.PARAMS['continue_on_error'] = False
    working_dir = os.path.join(work_dir, 'large_glacier_plot',
                               "{}_{}".format(model_type, gitype))
    cfg.PATHS['working_dir'] = utils.mkdir(working_dir, reset=True)
    gdirs = workflow.init_glacier_directories(gidf)

    task_list1 = [
        tasks.glacier_masks,
        tasks.compute_centerlines,
        tasks.initialize_flowlines,
        tasks.compute_downstream_line,
        tasks.compute_downstream_bedshape,
        tasks.catchment_area,
        tasks.catchment_intersections,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction,
        tasks.process_cru_data,
        tasks.local_t_star,
        tasks.mu_star_calibration,
        tasks.prepare_for_inversion]

    task_list2 = [tasks.filter_inversion_output,
                  tasks.distribute_thickness_per_altitude]

    for gdir in gdirs:
        if gdir.name == '1' and gitype == 'rgi':
            tasks.define_glacier_region(gdir, source='COPDEM')
        else:
            tasks.define_glacier_region(gdir, source='SRTM')
    for task in task_list1:
        execute_entity_task(task, gdirs)

    if model_type == 'oggm':
        execute_entity_task(tasks.mass_conservation_inversion, gdirs,
                            glen_a=cfg.PARAMS['inversion_glen_a']*p_value)
    elif model_type == 'glab':
        execute_entity_task(prepare_Glabtop_inputs, gdirs)
        execute_entity_task(base_stress_inversion, gdirs, calib_tal=p_value)

    for task in task_list2:
        execute_entity_task(task, gdirs)

    return gdirs


def comparing_glc_vol_in_large_glc():
    with open(os.path.join(cluster_dir, 'in_china_list.pkl'), 'rb') as f:
        in_cn_dict = pickle.load(f)
    in_cn_rgi = in_cn_dict['rgi_in_China']
    in_cn_ggi = in_cn_dict['ggi_in_China']
    glab_ggi = pd.read_csv(os.path.join(work_dir, 'Tienshan_glc_volume_uncertainty',
                                        'glab_srtm_ggi_opt.csv'))
    oggm_ggi = pd.read_csv(os.path.join(work_dir, 'Tienshan_glc_volume_uncertainty',
                                        'oggm_srtm_ggi_opt.csv'))
    glab_rgi = pd.read_csv(os.path.join(work_dir, 'Tienshan_glc_volume_uncertainty',
                                        'glab_srtm_rgi_opt.csv'))
    oggm_rgi = pd.read_csv(os.path.join(work_dir, 'Tienshan_glc_volume_uncertainty',
                                        'oggm_srtm_rgi_opt.csv'))
    glab_ggi_cn = glab_ggi[in_cn_ggi]
    glab_rgi_cn = glab_rgi[in_cn_rgi]
    oggm_ggi_cn = oggm_ggi[in_cn_ggi]
    oggm_rgi_cn = oggm_rgi[in_cn_rgi]
    oggm_rgi_cn100 = oggm_rgi_cn[oggm_rgi_cn.rgi_area_km2.astype(float)>100]
    oggm_ggi_cn100 = oggm_ggi_cn[oggm_ggi_cn.rgi_area_km2.astype(float)>100]
    (oggm_rgi_cn100.rgi_area_km2.astype(float).sum()-oggm_ggi_cn100.rgi_area_km2.astype(float).sum())/\
        oggm_ggi_cn100.rgi_area_km2.astype(float).sum()
    (oggm_rgi_cn100.inv_volume_km3.astype(float).sum()-oggm_ggi_cn100.inv_volume_km3.astype(float).sum())/\
    oggm_ggi_cn100.inv_volume_km3.astype(float).sum()


def calculate_mean_ice_thick():
    glab_ggi = pd.read_csv(os.path.join(work_dir, 'Tienshan_glc_volume_uncertainty',
                                        'glab_srtm_ggi_opt.csv'))
    oggm_ggi = pd.read_csv(os.path.join(work_dir, 'Tienshan_glc_volume_uncertainty',
                                        'oggm_srtm_ggi_opt.csv'))
    shean_mb_ts = pd.read_csv(os.path.join(data_dir, 'shean_mb.csv'))
    
ggidf = gpd.read_file(os.path.join(data_dir, 'volumed_gi', 'volumed_gamdam.shp'))
poly = Polygon([(79.53, 41.67), (79.53, 42.4), (80.64, 42.4), (80.64, 41.67), (79.53, 41.67)])
big_glc_region = gpd.GeoDataFrame(geometry=[poly], crs='epsg:4326')

rgidf = gpd.read_file(os.path.join(data_dir, 'larg_glc_plot', 'rgi'))
ggidf = gpd.read_file(os.path.join(data_dir, 'larg_glc_plot', 'ggi'))
China = gpd.read_file(os.path.join(root_dir, 'Data', 'cluster_output', 'china-shapefiles-master',
                                   'china.shp'))

# set up background
cfg.initialize()
base_dir = os.path.join(work_dir, 'large_glacier_plot')
working_dir = os.path.join(base_dir, 'base_map')
cfg.PATHS['working_dir'] = utils.mkdir(working_dir)
cfg.PARAMS['use_intersects'] = False
cfg.PARAMS['use_rgi_area'] = False
base_map_gdir = workflow.init_glacier_directories(glc_inventory2oggm(big_glc_region))[0]
tasks.define_glacier_region(base_map_gdir, source='SRTM')
tasks.glacier_masks(base_map_gdir)
dem_sm = salem.open_xr_dataset(base_map_gdir.get_filepath('dem'))
sm = salem.Map(dem_sm.salem.grid, countries=False, nx=1200)
sm.set_topography(base_map_gdir.get_filepath('dem'), relief_factor=.3)
cn_clip_mask = big_glc_region.copy()
cn_clip_mask['geometry'] = cn_clip_mask.buffer(0.05).values
sub_china = gpd.clip(China, cn_clip_mask)
#sm.set_shapefile(China, lw=2, ls=(0, (6, 2, 1, 2)), color='sienna', zorder=100)
sm.set_shapefile(China, lw=2, ls=(0, (6, 2, 1, 2)), color='dimgrey', zorder=100)
sm.set_lonlat_contours(xinterval=.4, yinterval=.3)
x, y = sm.grid.transform(80 + 6 / 60 + 59.11 / 3600, 42 + 2 / 60 + 13 / 46 / 3600)

# thickness/masks array prepare
ggidf_run = glc_inventory2oggm(ggidf.copy(), assigned_col_values={'Name': ggidf.name.values})
rgidf_run = rgidf.copy()

if os.path.exists(os.path.join(base_dir, 'gdirs.pkl')):
    with open(os.path.join(base_dir, 'gdirs.pkl'), 'rb') as f:
        gdirs = pickle.load(f)
    gdirs_rgi_oggm = gdirs['gdirs_rgi_oggm']
    gdirs_ggi_oggm = gdirs['gdirs_ggi_oggm']
    gdirs_rgi_glab = gdirs['gdirs_rgi_glab']
    gdirs_ggi_glab = gdirs['gdirs_ggi_glab']
else:
    gdirs_rgi_oggm = run_inversion_model_with_calib_param('oggm', 5.30, gidf=rgidf_run, gitype='rgi')
    gdirs_rgi_glab = run_inversion_model_with_calib_param('glab', 0.65, gidf=rgidf_run, gitype='rgi')
    gdirs_ggi_oggm = run_inversion_model_with_calib_param('oggm', 5.30, gidf=ggidf_run, gitype='ggi')
    gdirs_ggi_glab = run_inversion_model_with_calib_param('glab', 0.65, gidf=ggidf_run, gitype='ggi')
    gdirs = dict(gdirs_rgi_oggm=gdirs_rgi_oggm, gdirs_ggi_oggm=gdirs_ggi_oggm,
                 gdirs_rgi_glab=gdirs_rgi_glab, gdirs_ggi_glab=gdirs_ggi_glab)
    with open(os.path.join(base_dir, 'gdirs.pkl'), 'wb') as f:
        pickle.dump(gdirs, f)

dem_path = base_map_gdir.get_filepath('dem')
oggm_rgi_thick_arr, oggm_rgi_mask_arr = regional_thick_arr(dem_path, gdirs_rgi_oggm)
oggm_ggi_thick_arr, oggm_ggi_mask_arr = regional_thick_arr(dem_path, gdirs_ggi_oggm)
glab_rgi_thick_arr, glab_rgi_mask_arr = regional_thick_arr(dem_path, gdirs_rgi_glab)
glab_ggi_thick_arr, glab_ggi_mask_arr = regional_thick_arr(dem_path, gdirs_ggi_glab)
ref_glc = ['In', 'Ka', 'To', 'Ko']
other_glc = ['1', '2', '', '4']

sm_a = copy(sm)
sm_a.set_cmap('rainbow')
sm_a.set_data(np.where(oggm_ggi_mask_arr, oggm_ggi_thick_arr, np.nan))
sm_a.set_shapefile(ggidf_run[[name in ref_glc for name in ggidf_run.Name.values]], lw=.5)
sm_a.set_shapefile(ggidf_run[~np.array([name in ref_glc for name in ggidf_run.Name.values])], lw=0.5)
sm_a.set_plot_params(vmin=0, vmax=700)

sm_b = copy(sm)
sm_b.set_cmap('rainbow')
sm_b.set_data(np.where(glab_ggi_mask_arr, glab_ggi_thick_arr, np.nan))
sm_b.set_shapefile(ggidf_run[[name in ref_glc for name in ggidf_run.Name.values]], lw=.5)
sm_b.set_shapefile(ggidf_run[~np.array([name in ref_glc for name in ggidf_run.Name.values])], lw=0.5)
sm_b.set_plot_params(vmin=0, vmax=700)

sm_c = copy(sm)
sm_c.set_cmap('coolwarm')
oggm_rgi_ggi_mask = oggm_rgi_mask_arr + oggm_ggi_mask_arr
oggm_rgi_ggi_thick = oggm_rgi_thick_arr - oggm_ggi_thick_arr
sm_c.set_data(np.where(oggm_rgi_ggi_mask, oggm_rgi_ggi_thick, np.nan))
sm_c.set_shapefile(ggidf_run[[name in ref_glc for name in ggidf_run.Name.values]], lw=.5)
sm_c.set_shapefile(ggidf_run[~np.array([name in ref_glc for name in ggidf_run.Name.values])], lw=0.5)
sm_c.set_plot_params(vmin=-300, vmax=300)


sm_d = copy(sm)
sm_d.set_cmap('coolwarm')
glab_rgi_ggi_mask = glab_rgi_mask_arr + glab_ggi_mask_arr
glab_rgi_ggi_thick = glab_rgi_thick_arr - glab_ggi_thick_arr
sm_d.set_data(np.where(glab_rgi_ggi_mask, glab_rgi_ggi_thick, np.nan))
sm_d.set_shapefile(ggidf_run[[name in ref_glc for name in ggidf_run.Name.values]], lw=.5)
sm_d.set_shapefile(ggidf_run[~np.array([name in ref_glc for name in ggidf_run.Name.values])], lw=0.5)
sm_d.set_plot_params(vmin=-300, vmax=300)

fig, ([ax_a, ax_b], [ax_c, ax_d]) = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.02, hspace=0.012)
sm_a.visualize(ax=ax_a, addcbar=False)
sm_b.visualize(ax=ax_b, addcbar=False)
sm_c.visualize(ax=ax_c, addcbar=False)
sm_d.visualize(ax=ax_d, addcbar=False)
ax_a.scatter(x, y, marker='^', color='red', s=80,
           edgecolor='k', lw=2, zorder=150, label='Peak Tomur')
ax_b.scatter(x, y, marker='^', color='red', s=80,
             edgecolor='k', lw=2, zorder=150)
ax_c.scatter(x, y, marker='^', color='red', s=80,
             edgecolor='k', lw=2, zorder=150)
ax_d.scatter(x, y, marker='^', color='red', s=80,
             edgecolor='k', lw=2, zorder=150)
bbox_a = ax_a.get_position().bounds
bbox_b = ax_b.get_position().bounds
bbox_c = ax_c.get_position().bounds
bbox_d = ax_d.get_position().bounds

ax_a.set_position([bbox_a[0]-.03, bbox_a[1], bbox_a[2], bbox_a[3]])
ax_b.set_position([bbox_b[0]-.03, bbox_b[1], bbox_b[2], bbox_b[3]])
ax_c.set_position([bbox_c[0]-.03, bbox_c[1]+.04, bbox_c[2], bbox_c[3]])
ax_d.set_position([bbox_d[0]-.03, bbox_d[1]+.04, bbox_d[2], bbox_d[3]])
label_a = [a.get_text() for a in ax_a.get_yticklabels()]
ax_a.set_yticklabels(label_a, rotation=90, va='center')
label_c = [a.get_text() for a in ax_c.get_yticklabels()]
ax_c.set_yticklabels(label_c, rotation=90, va='center')
m = ax_a.scatter([0, 0], [1, 1], c=[0, 1000], cmap='rainbow',
                 s=0, vmin=0, vmax=700)
axc1 = fig.add_axes([0.88, 0.52, .01, .338])
label_color = [a.get_text() for a in axc1.get_yticklabels()]
axc1.set_yticklabels(label_color, rotation=90, va='center')
fig.colorbar(m, axc1, label='Ice thickness (m)', extend='max', ticks=range(50, 701, 150))

m = ax_a.scatter([0, 0], [1, 1], c=[-300, 300], cmap='coolwarm',
                 s=0, vmin=-300, vmax=300)
label_c = [a.get_text() for a in ax_d.get_yticklabels()]
axc2 = fig.add_axes([0.88, 0.171, .01, .338])
label_color = [a.get_text() for a in axc2.get_yticklabels()]
axc2.set_yticklabels(label_color, rotation=90, va='center')
fig.colorbar(m, axc2, label='Ice thickness difference (m)', extend='both', ticks=[-240, -120, 0, 120, 240])

x_ka, y_ka = sm.grid.transform(79.72109, 42.07352)
x_to, y_to = sm.grid.transform(80.00131, 41.93178)
x_ko, y_ko = sm.grid.transform(80.10628, 41.80270)
x_in, y_in = sm.grid.transform(80.04467, 42.12272)
#for ax in [ax_a, ax_b, ax_c, ax_d]:
#    ax.text(x_ka-200, y_ka+60, s='Ka', weight='bold', fontsize=10, color='lightgrey',
#              bbox=dict(boxstyle='square', fc='grey', ec='dimgrey'))
#    ax.text(x_to-230, y_to+200, s='To', weight='bold', fontsize=10, color='lightgrey',
#              bbox=dict(boxstyle='square', fc='grey', ec='dimgrey'))
#    ax.text(x_ko-50, y_ko+160, s='Ko', weight='bold', fontsize=10, color='lightgrey',
#              bbox=dict(boxstyle='square', fc='grey', ec='dimgrey'))
#    ax.text(x_in-450, y_in-130, s='In', weight='bold', fontsize=10, color='lightgrey',
#              bbox=dict(boxstyle='square', fc='grey', ec='dimgrey'))

ax_a.text(x_ka-200, y_ka+60, s='Ka', weight='bold', fontsize=10, color='lightgrey',
            bbox=dict(boxstyle='square', fc='grey', ec='dimgrey'))
ax_a.text(x_to-230, y_to+200, s='To', weight='bold', fontsize=10, color='lightgrey',
            bbox=dict(boxstyle='square', fc='grey', ec='dimgrey'))
ax_a.text(x_ko-50, y_ko+160, s='Ko', weight='bold', fontsize=10, color='lightgrey',
            bbox=dict(boxstyle='square', fc='grey', ec='dimgrey'))
ax_a.text(x_in-450, y_in-130, s='In', weight='bold', fontsize=10, color='lightgrey',
            bbox=dict(boxstyle='square', fc='grey', ec='dimgrey'))

ax_a.text(.01, .99, ha='left', va='top', s='A', weight='bold', fontsize=10, transform=ax_a.transAxes)
ax_b.text(.01, .99, ha='left', va='top', s='B', weight='bold', fontsize=10, transform=ax_b.transAxes)
ax_c.text(.01, .99, ha='left', va='top', s='C', weight='bold', fontsize=10, transform=ax_c.transAxes)
ax_d.text(.01, .99, ha='left', va='top', s='D', weight='bold', fontsize=10, transform=ax_d.transAxes)
ax_a.bar(0, 0, lw=.4, ec='black', fc='none', label='GGI Outline')
ax_a.bar(0, 0, ls=(0, (6, 2, 1, 2)), lw=2, ec='dimgrey', fc='none', label='International\nBoundary')
ax_a.legend(loc='lower right', prop=dict(weight='bold', size=8.5), labelspacing=1)
fig.savefig(os.path.join(data_dir, 'figure', 'glacier_near_tuomuer.pdf'), dpi=300, bbox_inches='tight')

rgidf = gpd.read_file(os.path.join(data_dir, 'larg_glc_plot', 'rgi'))
ggidf = gpd.read_file(os.path.join(data_dir, 'larg_glc_plot', 'ggi'))
oggm_inv_df = pd.read_csv(os.path.join(root_dir, 'Data', 'model_output', 'Tienshan_glc_volume_uncertainty',
                                       'oggm_srtm_ggi_opt.csv'))
glab_inv_df = pd.read_csv(os.path.join(root_dir, 'Data', 'model_output', 'Tienshan_glc_volume_uncertainty',
                                       'glab_srtm_ggi_opt.csv'))
oggm_inv_df.index = oggm_inv_df.rgi_id
glab_inv_df.index = glab_inv_df.rgi_id
fr_vol = gpd.read_file(os.path.join(data_dir, 'volumed_gi', 'volumed_rgi.shp'))
ggidf.index = ggidf.rgi_id
fr_vol.index = fr_vol.RGIId
fr_vol = fr_vol.loc[rgidf.RGIId]
glc_name = ['In', 'To', 'Ka', 'Ko']
rgi_tino_id, ggi_tino_id = [], []
for gname in glc_name:
    rgi_tino_id.append(rgidf[rgidf.Name==gname].RGIId.values[0])
    ggi_tino_id.append(ggidf[ggidf.name==gname].rgi_id.values[0])
m0_vol = []
for rid in rgi_tino_id:
    path = get_farinotti_file_path(rid, model_type='0')
    ds = salem.open_xr_dataset(path)
    vol = ds.data.values.sum() * ds.salem.grid.dx**2 * 1e-9
    m0_vol.append(vol)
rgidf.index = rgidf.RGIId
rgi_tino_area = rgidf.loc[rgi_tino_id].Area.values
glab_inv_df = glab_inv_df.loc[ggi_tino_id]
oggm_inv_df = oggm_inv_df.loc[ggi_tino_id]
oggm_tino_thick = (oggm_inv_df.inv_volume_km3 / oggm_inv_df.rgi_area_km2 * 1e3).values
glab_tino_thick = (glab_inv_df.inv_volume_km3 / glab_inv_df.rgi_area_km2 * 1e3).values
tino_thick = [132.3, 90.4, 77.4, 88.6]
m0 = m0_vol / rgi_tino_area * 1e3
m1 = [82.27, 44.47, 11.35, 10.02] / rgi_tino_area * 1e3
m2 = [63.59, 29.20, 10.71, 8.22] / rgi_tino_area * 1e3
m3 = [87.27, 63.49, 11.24, 13.45] / rgi_tino_area * 1e3
m4 = [26.03, 18.04, 4.74, 2.66] / rgi_tino_area * 1e3
hf = [233, 127, 140, 145]
lins = [200.9, 136.0, 129.8, 134.1]
su = [329.4, 261.6, 193.7, 179.0]


import seaborn as sns
import matplotlib.pyplot as plt
df = pd.DataFrame(dict(thickness=np.append(tino_thick, [oggm_tino_thick, glab_tino_thick, m0, m1, m2, m3, m4]),
                       method= ['P2018']*4+['MC']*4+['BS']*4+['Fcons.']*4+['F1']*4+['F2']*4+['F3']*4+['F4']*4,
                       glc=glc_name*8))
my_pal = {'P2018':'mediumpurple', 'MC': 'skyblue', 'BS': 'lightcoral', 'Fcons.': 'navajowhite', 'F1': 'gray', 
          'F2': 'silver', 'F3': 'lightgrey', 'F4': 'whitesmoke'}
fig, ax = plt.subplots(figsize=(4, 3))
ax = sns.barplot(x='glc', y='thickness', hue='method', data=df, palette=my_pal, ec='#535353', lw=1.5)

ax.legend(ncol=3)

df1 = pd.DataFrame(dict(thickness=np.append(tino_thick, [oggm_tino_thick,
                                                         glab_tino_thick,
                                                         hf, lins, su]),
                        method=['P2018']*4+['MC']*4+['BS']*4+['Huss']*4+['Lins']*4+['Su']*4,
                        glc=glc_name*6))
my_pal1 = {'P2018':'mediumpurple',
           'MC': 'skyblue',
           'BS': 'lightcoral',
           'Huss': 'gray',
           'Lins': 'silver', 
           'Su': 'whitesmoke'}
fig, ax = plt.subplots(figsize=(5, 3))
ax = sns.barplot(x='glc', y='thickness', hue='method', data=df1,
                 palette=my_pal1, ec='#535353', lw=1.5)
ax.set_xticklabels(['South Inylchek', 'Tomur', 'Kaindy', 'Koxkar'])
ax.set_ylabel('Mean thickness (m)')
ax.set_xlabel('Glacier')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=[handles[0], handles[3], handles[1], handles[4], handles[2], handles[5]],
          labels=[labels[0], labels[3], labels[1], labels[4], labels[2], labels[5]],
          ncol=3)
plt.tight_layout()
fig.savefig(os.path.join(data_dir, 'figure', 'compare_with_Pieczonka2018.pdf'), dpi=300,
            bbox_inches='tight')
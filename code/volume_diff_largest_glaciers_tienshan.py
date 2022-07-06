#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 21:10:00 2020

@author: keeptg
"""

from oggm import tasks, utils, graphics, workflow, cfg
from oggm.graphics import OGGM_CMAPS
import warnings
warnings.filterwarnings('ignore', 'GeoSeries.notna', UserWarning)

import os, pickle
from tienshan_glacier_thick import run_model
from Re_GlabTop import glc_inventory2oggm

import numpy as np
import pandas as pd

from shapely.geometry import Point, Polygon
import geopandas as gpd

import salem
from salem import sample_data_dir

#%matplotlib widget
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

from path_config import *

def find_gdirs(target_rgiid, target_ggiid):

    for oggm_gam_gdir in oggm_gam1_gdirs:
        if oggm_gam_gdir.rgi_id==target_ggiid:
            break
    else:
        oggm_gam_gdir = None
    
    for glab_gam_gdir in glab_gam1_gdirs:
        if glab_gam_gdir.rgi_id==target_ggiid:
            break
    else:
        glab_gam_gdir = None
    
    for oggm_rgi_gdir in oggm_rgi1_gdirs:
        if oggm_rgi_gdir.rgi_id==target_rgiid:
            break
    else:
        oggm_rgi_gdir = None
        
    for glab_rgi_gdir in glab_rgi1_gdirs:
        if glab_rgi_gdir.rgi_id == target_rgiid:
            break
    else:
        glab_rgi_gdir = None
        
    return oggm_gam_gdir, glab_gam_gdir, oggm_rgi_gdir, glab_rgi_gdir

def get_topo_gdirs(geo_concen_date, working_dir, reset=True):

    if not reset and os.path.exists(os.path.join(working_dir, 'loc_gdir.pkl')):
        with open(os.path.join(working_dir, 'loc_gdir.pkl'), 'rb') as f:
            loc_gdir = pickle.load(f)
    else:
        geom_names = [geom for geom in geo_concen_date.columns if 'geometry' in geom]
        maxx, maxy, minx, miny = [], [], [], []
        for geom_name in geom_names:
            reset_geo_concen = geo_concen_date.copy().set_geometry(geom_name)
            maxx.append(reset_geo_concen.geometry.bounds.maxx.max())
            maxy.append(reset_geo_concen.geometry.bounds.maxy.max())
            minx.append(reset_geo_concen.geometry.bounds.minx.min())
            miny.append(reset_geo_concen.geometry.bounds.miny.min())
            
        maxx = np.max(maxx)
        maxy = np.max(maxy)
        minx = np.min(minx)
        miny = np.min(miny)
        geom = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny),
                        (minx, miny)])
        loc_gdf = gpd.GeoDataFrame(geometry=[geom], crs='epsg:4326')
        
        loc_gdf = glc_inventory2oggm(loc_gdf)
        wd = os.path.join(working_dir, 'loc_gdf')
        utils.mkdir(wd, reset=True)
        cfg.initialize()
        cfg.PATHS['working_dir'] = wd
        cfg.PARAMS['use_rgi_area'] = False
        cfg.PARAMS['use_intersects'] = False
        cfg.PARAMS['dmax'] = 90.
        loc_gdir = workflow.init_glacier_directories(loc_gdf)[0]
        tasks.define_glacier_region(loc_gdir, source='SRTM')
        with open(os.path.join(working_dir, 'loc_gdir.pkl'), 'wb') as f:
            pickle.dump(loc_gdir, f)
    
    return loc_gdir


def get_thick_and_diff(gdir, geo_df):

    demp = gdir.get_filepath('dem')
    dem = salem.open_xr_dataset(demp)
    gdir = dem.salem.grid
    ortarr = np.zeros(dem.data.shape)
    grtarr = np.zeros(dem.data.shape)
    ogtarr = np.zeros(dem.data.shape)
    ggtarr = np.zeros(dem.data.shape)
    rmarr = np.zeros(dem.data.shape)
    gmarr = np.zeros(dem.data.shape)
    geo_concen_date = geo_df
    for i in range(len(geo_concen_date)):
        rgiid = geo_concen_date.iloc[[i]].rgi_id.values[0]
        ggiid = geo_concen_date.iloc[[i]].ggi_id.values[0]
        ogg, ggg, org, grg = find_gdirs(rgiid, ggiid)
        gd = salem.open_xr_dataset(ogg.get_filepath('gridded_data'))
        ogt = gd.distributed_thickness
        ogm = gd.glacier_mask
        gd = salem.open_xr_dataset(ggg.get_filepath('gridded_data'))
        ggt = gd.distributed_thickness
        gd = salem.open_xr_dataset(org.get_filepath('gridded_data'))
        try:
            ort = gd.distributed_thickness
        except AttributeError:
            continue
        orm = gd.glacier_mask
        gd = salem.open_xr_dataset(grg.get_filepath('gridded_data'))
        grt = gd.distributed_thickness
        
        ortp = dem.salem.transform(ort)
        ortarr += np.where(np.isnan(ortp.data), 0, ortp.data)
        grtp = dem.salem.transform(grt)
        grtarr += np.where(np.isnan(grtp.data), 0, grtp.data)
        ogtp = dem.salem.transform(ogt)
        ogtarr += np.where(np.isnan(ogtp.data), 0, ogtp.data)
        ggtp = dem.salem.transform(ggt)
        ggtarr += np.where(np.isnan(ggtp.data), 0, ggtp.data)
        
        ogmp = dem.salem.transform(ogm)
        gmarr += np.where(np.isnan(ogmp.data), 0, ogmp.data)
        ormp = dem.salem.transform(orm)
        rmarr += np.where(np.isnan(ormp.data), 0, ormp.data)
        
    return ortarr, ogtarr, grtarr, ggtarr, rmarr, gmarr


def get_cline_oline(geo_concen_date):
    cr_list, cg_list, or_list, og_list = [], [], [], []
    for i in range(len(geo_concen_date)):
        rid = geo_concen_date.iloc[[i]].rgi_id
        gid = geo_concen_date.iloc[[i]].ggi_id
        gdir5_or = [gdir for gdir in oggm_rgi1_gdirs if 
                    gdir.rgi_id==rid.values[0]][0]
        gdir5_og = [gdir for gdir in oggm_gam1_gdirs if 
                    gdir.rgi_id==gid.values[0]][0]
        utils.write_centerlines_to_shape([gdir5_or],
                                         path=os.path.join(gdir5_or.dir,
                                                            'centerline.shp'))
        utils.write_centerlines_to_shape([gdir5_og], 
                                         path=os.path.join(gdir5_og.dir,
                                                           'centerline.shp'))
        cline5_or = gpd.read_file(os.path.join(gdir5_or.dir, 'centerline.shp'))
        cline5_og = gpd.read_file(os.path.join(gdir5_og.dir, 'centerline.shp'))
        oline5_or = gdir5_or.read_shapefile('outlines')
        oline5_og = gdir5_og.read_shapefile('outlines')
        oline5_or.to_crs("EPSG:4326", inplace=True)
        oline5_og.to_crs("EPSG:4326", inplace=True)
        cr_list.append(cline5_or), cg_list.append(cline5_og)
        or_list.append(oline5_or), og_list.append(oline5_og)

    return cr_list, cg_list, or_list, og_list


def thickdiff_in_largest_glc_plot(savefig=False, reset=False):

    global oggm_rgi1_gdirs, oggm_gam1_gdirs, glab_rgi1_gdirs, glab_gam1_gdirs
    if run_in_cluster:
        working_dir = work_dir
    else:
        working_dir = os.path.join(root_dir, 'Data', 'model_output', 
                                'largest_glc_in_tienshan')
    cfg.initialize()

    dir_ = os.path.join(root_dir, 'cluster_output')

    # GAM Data
    oggm_gam1 = pd.read_csv(os.path.join(dir_, 
                                        'tienshan_ice_volume', 
                                        'Tienshan_glc_volume_uncertainty',
                                        'oggm_ggi_SRTM_opt(5.30).csv'))
    oggm_gam1.sort_values(by='rgi_area_km2', ascending=True, inplace=True)
    #glab_gam1 = pd.read_csv(os.path.join(dir_,
    #                                    'tienshan_ice_volume',
    #                                    'glacier_statistics_glab_gam_minslope5deg.csv'))

    # RGI Data
    oggm_rgi1 = pd.read_csv(os.path.join(dir_,
                                        'tienshan_ice_volume',
                                        'Tienshan_glc_volume_uncertainty',
                                        'oggm_rgi_SRTM_opt(5.30).csv'))
    oggm_rgi1.sort_values(by='rgi_area_km2', ascending=True, inplace=True)
    #oggm_gam1 = pd.read_csv(os.path.join(dir_,
    #                                    'tienshan_ice_volume',
    #                                    'glacier_statistics_glab_gam_minslope5deg.csv'))

    rgidf = gpd.read_file(os.path.join(data_dir, 'rgi'))
    ggidf = gpd.read_file(os.path.join(data_dir, 'checked_gamdam'))
    rgidf.index = rgidf.RGIId
    ggidf.index = ggidf.RGIId

    rgidf = rgidf.loc[oggm_rgi1.rgi_id.iloc[-2:].values.tolist()].copy()
    ggidf = ggidf.loc[oggm_gam1.rgi_id.iloc[-2:].values.tolist()].copy()

    if (not reset) and os.path.exists(os.path.join(working_dir, 'gdirs.pkl')):
        with open(os.path.join(working_dir, 'gdirs.pkl'), 'rb') as f:
            gdirs_dict = pickle.load(f)
        oggm_rgi1_gdirs = gdirs_dict['oggm_rgi1_gdirs']
        oggm_gam1_gdirs = gdirs_dict['oggm_gam1_gdirs']
        glab_rgi1_gdirs = gdirs_dict['glab_rgi1_gdirs']
        glab_gam1_gdirs = gdirs_dict['glab_gam1_gdirs']
    else:
        #dir_ = os.path.join(root_dir, 'Data', 'cluster_output')

        glab_gam1_gdirs = run_model(ggidf, 'glabtop',
                                    os.path.join(working_dir, 'glab_ggi'),
                                    resetworking_dir=reset,
                                    totle_task=reset,
                                    calib_tal=0.65,
                                    dem_type='SRTM',)

        glab_rgi1_gdirs = run_model(rgidf, 'glabtop',
                                    os.path.join(working_dir, 'glab_rgi'),
                                    resetworking_dir=reset,
                                    totle_task=reset,
                                    dem_type='SRTM',
                                    calib_tal=0.65)

        oggm_gam1_gdirs = run_model(ggidf, 'oggm',
                                    os.path.join(working_dir, 'oggm_ggi'),
                                    resetworking_dir=reset,
                                    totle_task=reset,
                                    dem_type='SRTM',
                                    glen_a=5.30*2.4*1e-24)

        oggm_rgi1_gdirs = run_model(rgidf, 'oggm',
                                    os.path.join(working_dir, 'oggm_rgi'),
                                    resetworking_dir=reset,
                                    totle_task=reset,
                                    dem_type='SRTM',
                                    glen_a=5.30*2.4*1e-24)
        gdirs_dict = {'oggm_rgi1_gdirs': oggm_rgi1_gdirs,
                      'oggm_gam1_gdirs': oggm_gam1_gdirs,
                      'glab_rgi1_gdirs': glab_rgi1_gdirs,
                      'glab_gam1_gdirs': glab_gam1_gdirs}
        with open(os.path.join(working_dir, 'gdirs.pkl'), 'wb') as f:
            pickle.dump(gdirs_dict, f)

    glab_gam1_output = utils.compile_glacier_statistics(glab_gam1_gdirs)
    glab_rgi1_output = utils.compile_glacier_statistics(glab_rgi1_gdirs)
    oggm_gam1_output = utils.compile_glacier_statistics(oggm_gam1_gdirs)
    oggm_rgi1_output = utils.compile_glacier_statistics(oggm_rgi1_gdirs)

    concen_date = pd.DataFrame(columns=['rgi_id', 'ggi_id', 'rgi_area', 'ggi_area', 
                                        'oggm_rgi_v', 'oggm_ggi_v', 
                                        'glab_rgi_v', 'glab_ggi_v',
                                        'rgi_geometry', 'ggi_geometry'])
    concen_date['rgi_id'] = oggm_rgi1_output.index
    concen_date['rgi_area'] = oggm_rgi1_output.rgi_area_km2.values
    concen_date['oggm_rgi_v'] = oggm_rgi1_output.inv_volume_km3.values
    concen_date['glab_rgi_v'] = glab_rgi1_output.inv_volume_km3.values
    concen_date['rgi_geometry'] = rgidf.geometry.values

    concen_date['ggi_area'] = oggm_gam1_output.rgi_area_km2
    concen_date['oggm_ggi_v'] = oggm_gam1_output.inv_volume_km3
    concen_date['glab_ggi_v'] = glab_gam1_output.inv_volume_km3
    concen_date['ggi_geometry'] = ggidf.geometry.values
    concen_date['ggi_id'] = oggm_gam1_output.index

    geo_concen_date = gpd.GeoDataFrame(concen_date, crs='epsg:4326', 
                                    geometry='rgi_geometry')

    geo_concen_date.sort_values(by='rgi_area', inplace=True)


    choosed_geo_concen_date = geo_concen_date
    gdir = get_topo_gdirs(choosed_geo_concen_date, working_dir, reset=reset)
    grid = gdir.grid
    demp = gdir.get_filepath('dem')
    ortarr, ogtarr, grtarr, ggtarr, rmarr, gmarr = \
        get_thick_and_diff(gdir, choosed_geo_concen_date)
    cr_list, cg_list, or_list, og_list = \
        get_cline_oline(choosed_geo_concen_date)
    cline_color = ['darkred', 'deepskyblue']
    cline_main_color = 'darkorange'
    cline_sub_color = 'dimgrey'

    # Peak Pobeda geo-coordination
    x = 80 + 6 / 60 + 54.35 / 3600
    y = 42 + 2 / 60 + 3.4 / 3600
    xt, yt = grid.transform([x], [y])

    sm_outline_width = .5
    sm_cline_width = 1.2
    diff_vmax = 240
    legend_fontsize = 7
    sm1 = salem.Map(grid, nx=1300, countries=False)
    sm1.set_topography(demp, relief_factor=.3)
    sm1.set_data(np.where(gmarr+rmarr>0, ortarr-ogtarr, np.nan))
    sm1.set_cmap('coolwarm')
    sm1.set_plot_params(vmin=-diff_vmax, vmax=diff_vmax)
    for i in range(len(choosed_geo_concen_date)):
        sm1.set_shapefile(og_list[i], edgecolor='black', linewidth=sm_outline_width,
                        facecolor='none', )
        sm1.set_lonlat_contours(xinterval=.2, zorder=1)

    sm2 = salem.Map(grid, nx=1300, countries=False)
    sm2.set_topography(demp, relief_factor=.3)
    sm2.set_data(np.where(gmarr+rmarr>0, ggtarr-grtarr, np.nan))
    sm2.set_cmap('coolwarm')
    sm2.set_plot_params(vmin=-diff_vmax, vmax=diff_vmax)
    for i in range(len(choosed_geo_concen_date)):
        sm2.set_shapefile(og_list[i], edgecolor='black', linewidth=sm_outline_width,
                        facecolor='none', )
        sm2.set_lonlat_contours(xinterval=.2, zorder=1)

    smap_r = salem.Map(gdir.grid, countries=False, nx=1300)
    for cline in cr_list:
        smap_r.set_shapefile(cline.iloc[1:], linewidth=sm_cline_width, color=cline_sub_color, 
                            zorder=100)
        smap_r.set_shapefile(cline.iloc[[0]], linewidth=sm_cline_width, color=cline_main_color,
                            zorder=100)
    smap_g = salem.Map(gdir.grid, countries=False, nx=1300)
    for cline in cg_list:
        smap_g.set_shapefile(cline.iloc[1:], linewidth=1.5, color=cline_sub_color,
                            zorder=100)
        smap_g.set_shapefile(cline.iloc[[0]], linewidth=1.5, color=cline_main_color,
                            zorder=100)

    fig, axs = plt.subplots(3, 2, figsize=(8, 10.1), sharey=True, sharex=True)
    rids = choosed_geo_concen_date['rgi_id'].values
    gids = choosed_geo_concen_date['ggi_id'].values
    oggm_gam_gdirs, glab_gam_gdirs, oggm_rgi_gdirs, glab_rgi_gdirs = \
        [], [], [], []
    for i in range(len(choosed_geo_concen_date)):
        rid = choosed_geo_concen_date.iloc[i]['rgi_id']
        gid = choosed_geo_concen_date.iloc[i]['ggi_id']
        oggm_gam_gdir, glab_gam_gdir, oggm_rgi_gdir, glab_rgi_gdir =\
            find_gdirs(rid, gid)
        oggm_gam_gdirs.append(oggm_gam_gdir), glab_gam_gdirs.append(glab_gam_gdir)
        oggm_rgi_gdirs.append(oggm_rgi_gdir), glab_rgi_gdirs.append(glab_rgi_gdir)

    vmax = 600
    graphics.plot_inversion(oggm_gam_gdirs, smap=smap_g, title='', vmax=vmax,
                            add_colorbar=False, ax=axs[0, 0], add_scalebar=False,
                            lonlat_contours_kwargs=dict(alpha=0))
    graphics.plot_inversion(glab_gam_gdirs, smap=smap_g, title='', vmax=vmax,
                            add_colorbar=False, ax=axs[0, 1], add_scalebar=False,
                            lonlat_contours_kwargs=dict(alpha=0))
    graphics.plot_inversion(oggm_rgi_gdirs, smap=smap_r, title='', vmax=vmax,
                            add_colorbar=False, ax=axs[1, 0], add_scalebar=False,
                            lonlat_contours_kwargs=dict(alpha=0))
    graphics.plot_inversion(glab_rgi_gdirs, smap=smap_r, title='', vmax=vmax,
                            add_colorbar=False, ax=axs[1, 1], add_scalebar=False,
                            lonlat_contours_kwargs=dict(alpha=0))
    ax2, ax3 = axs[2, 0], axs[2, 1]
    m = ax3.scatter((0, 0), (0, 0), s=0, c=[10, 20], vmin=-240, vmax=240,
                    cmap='coolwarm')
    sm1.visualize(ax=ax2, addcbar=False)
    sm2.visualize(ax=ax3, addcbar=False)
    #raise TypeError("Avoid axes size unormal Error!")

    # Move axes
    for ax in axs[:,0]:
        bbox = ax.get_position()
        ax.set_position([bbox.x0-.03, bbox.y0, bbox.width, bbox.height])
        bbox = ax.get_position()
    for ax in axs[:,1]:
        bbox = ax.get_position()
        ax.set_position([bbox.x0-.08, bbox.y0, bbox.width, bbox.height])
        bbox = ax.get_position()

    plt.tight_layout()
    plt.subplots_adjust(wspace=.03, hspace=-.2, right=.89)
    axs[0, 0].set_title('Mass conservation model')
    axs[0, 1].set_title('Basal shear stress model')
    # Add colorbar
    m1 = axs[1, 1].scatter((0, 0), (0, 0), s=0, c=[10, 20], vmin=0, vmax=vmax,
                        cmap=OGGM_CMAPS['section_thickness'])
    bbox1, bbox2 , bbox3 = axs[0, 1].get_position(), axs[1, 1].get_position(), axs[2, 1].get_position()
    cax1 = fig.add_axes([bbox1.x1+.013, bbox2.y0, 0.01, bbox1.y1-bbox2.y0])
    fig.colorbar(m1, cax1, extend='max', label='Ice thickness (m)')

    m = ax3.scatter((0, 0), (0, 0), s=0, c=[10,20], vmin=-240, vmax=240,
                    cmap='coolwarm')
    cax2 = fig.add_axes([bbox3.x1+.013, bbox3.y0, 0.01, bbox3.height])
    fig.colorbar(m, cax2, extend='both', label='Ice thickness difference (m)')

    axs[1, 1].scatter((0, 0), (0, 0), s=0, c=[10, 20], vmin=0, vmax=vmax,
                    cmap=OGGM_CMAPS['section_thickness'])

    # Add legend
    ax2.bar(0, 0, 0, linewidth=sm_outline_width, ec='black', fc='none', label='GAMDAMv2\nOutline')
    ax2.legend(loc=3, labelspacing=.6, handlelength=1, handletextpad=.3,
               prop=dict(weight='bold'), fontsize=legend_fontsize)

    ax3.bar(0, 0, 0, linewidth=sm_outline_width, ec='black', fc='none', label='GAMDAMv2\nOutline')
    ax3.legend(loc=3, labelspacing=.6, handlelength=1,
               handletextpad=.3, prop=dict(weight='bold'), fontsize=legend_fontsize)

    ax=axs[0, 0]
    ax.bar(0, 0, 0, linewidth=1.5, ec='k', fc='none', label='GAMDAMv2\nOutline')
    ax.plot((0, 0), (0, 0), linewidth=1.2, color=cline_main_color, 
            label='Main\nFlowline')
    ax.plot((0, 0), (0, 0), linewidth=1.2, color=cline_sub_color, 
            label='Tributary\nFlowline')
    ax.legend(loc=3, labelspacing=.6, handlelength=1, handletextpad=.3,
              prop=dict(weight='bold'), fontsize=legend_fontsize)

    ax=axs[1, 0]
    ax.bar(0, 0, 0, linewidth=1.5, ec='k', fc='none', label='RGIv6\nOutline')
    ax.plot((0, 0), (0, 0), linewidth=1.2, color=cline_main_color, 
            label='Main\nFlowline')
    ax.plot((0, 0), (0, 0), linewidth=1.2, color=cline_sub_color, 
            label='Tributary\nFlowline')
    ax.legend(loc=3, labelspacing=.6, handlelength=1, handletextpad=.3,
              prop=dict(weight='bold'), fontsize=legend_fontsize)

    ax=axs[0, 1]
    ax.bar(0, 0, 0, linewidth=1.5, ec='k', fc='none', label='GAMDAMv2\nOutline')
    ax.plot((0, 0), (0, 0), linewidth=1.2, color=cline_main_color, 
            label='Main\nFlowline')
    ax.plot((0, 0), (0, 0), linewidth=1.2, color=cline_sub_color, 
            label='Tributary\nFlowline')
    ax.legend(loc=3, labelspacing=.6, handlelength=1, handletextpad=.3,
              prop=dict(weight='bold'), fontsize=legend_fontsize)

    ax=axs[1, 1]
    ax.bar(0, 0, 0, linewidth=1.5, ec='k', fc='none', label='RGIv6\nOutline')
    ax.plot((0, 0), (0, 0), linewidth=1.2, color=cline_main_color, 
            label='Main\nFlowline')
    ax.plot((0, 0), (0, 0), linewidth=1.2, color=cline_sub_color, 
            label='Tributary\nFlowline')
    ax.legend(loc=3, labelspacing=.6, handlelength=1, handletextpad=.3,
              prop=dict(weight='bold'), fontsize=legend_fontsize)

    # Add glacier name
    ax = axs[2, 0]
    ax.text(.2, .94, "South Inylchek Glacier", color='k', ha='left',
            va='top', weight='bold', transform=ax.transAxes, fontsize=9)
    ax.text(.97, .03, "Tomur Glacier", color='k', ha='right',
            va='bottom', weight='bold', transform=ax.transAxes, fontsize=9)

    ax = axs[2, 1]
    ax.text(.2, .94, "South Inylchek Glacier", color='k', ha='left',
            va='top', weight='bold', transform=ax.transAxes, fontsize=9)
    ax.text(.97, .03, "Tomur Glacier", color='k', ha='right',
            va='bottom', weight='bold', transform=ax.transAxes, fontsize=9)

    # Add annotate
    ax = axs[2, 0]
    ax.text(.86, .35, 'Over-\nestimate', weight='bold', color='red', ha='center',
            va='center', transform=ax.transAxes, style='italic')
    ax.arrow(.85, .41, -.12, .05, transform=ax.transAxes, width=.005, color='red')
    ax.arrow(.8, .3, -.03, -.05, transform=ax.transAxes, width=.005, color='red')

    ax.text(.25, .38, 'Under-\nestimate', weight='bold', color='blue', ha='center',
            va='center', transform=ax.transAxes, style='italic')
    ax.arrow(.36, .4, .09, -.03, transform=ax.transAxes, width=.005, color='blue')
    ax.arrow(.35, .33, .05, -.06, transform=ax.transAxes, width=.005, color='blue')

    ax.text(.15, .23, 'Less\ndifference', weight='bold', color='lightseagreen', ha='center',
            va='center', transform=ax.transAxes, style='italic')
    ax.arrow(.23, .25, .1, -.05, transform=ax.transAxes, width=.005, color='lightseagreen')
    ax.arrow(.28, .18, .03, -.06, transform=ax.transAxes, width=.005, color='lightseagreen')

    # Add Axes label
    label_list = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, ax in enumerate(np.array(axs).flatten()):
        label = label_list[i]
        ax.text(.019, .98, label, ha='left', va='top', 
                transform=ax.axes.transAxes,
    #            bbox={'fc':'grey', 'boxstyle':'square', 'alpha':.4,
    #                  'ec':'dimgrey', 'lw':1.5},
                fontweight='bold')

    # Save figure
    if savefig:
        if run_in_cluster:
            path = os.path.join(out_dir, 'figure')
        else:
            path = os.path.join(data_dir, 'figure')
        utils.mkdir(path)
        fig.savefig(os.path.join(path, 'thickdiff_in_largest_glc.pdf'),
                    dpi=300, bbox_inches='tight')
    

def glaciers_in_boundary_of_china_plot(use_google_map=False, savefig=False):
    
    import copy
    dir_ = os.path.join(root_dir, 'cluster_output')
    china_gdf = gpd.read_file(os.path.join('/home/lifei/Data', 'Shpfile_repository', 
                                           'china-shapefiles', 'shapefiles',
                                           'china.shp'))
    rgidf = gpd.read_file(os.path.join(data_dir, 'rgi'))
    ggidf = gpd.read_file(os.path.join(data_dir, 'checked_gamdam'))
    rgidf.index = rgidf.RGIId
    ggidf.index = ggidf.RGIId
    oggm_gam1 = pd.read_csv(os.path.join(dir_, 
                                        'tienshan_ice_volume', 
                                        'Tienshan_glc_volume_uncertainty',
                                        'oggm_ggi_SRTM_opt(5.30).csv'))
    oggm_gam1.sort_values(by='rgi_area_km2', ascending=True, inplace=True)
    oggm_rgi1 = pd.read_csv(os.path.join(dir_,
                                        'tienshan_ice_volume',
                                        'Tienshan_glc_volume_uncertainty',
                                        'oggm_rgi_SRTM_opt(5.30).csv'))
    oggm_rgi1.sort_values(by='rgi_area_km2', ascending=True, inplace=True)

    rgidf = rgidf.loc[oggm_rgi1.rgi_id.iloc[-2:].values.tolist()].copy()
    ggidf = ggidf.loc[oggm_gam1.rgi_id.iloc[-2:].values.tolist()].copy()
    if run_in_cluster:
        working_dir = work_dir
    else:
        working_dir = os.path.join(root_dir, 'Data', 'model_output', 
                                'largest_glc_in_tienshan')
    gdir = get_topo_gdirs(pd.concat([rgidf, ggidf]), working_dir, reset=False)
    grid = gdir.grid
    demp = gdir.get_filepath('dem')
    # Peak Tomur geo-coordination
    sm_cbline_width = 2
    sm = salem.Map(grid, nx=1300, countries=False)
    x, y = sm.grid.transform(80 + 6 / 60 + 59.11 / 3600, 42 + 2 / 60 + 13 / 46 / 3600)
    if use_google_map:
        g = salem.GoogleVisibleMap(x=sm.grid.extent_in_crs('epsg:4326')[:2],
                                y=sm.grid.extent_in_crs('epsg:4326')[2:],
                                crs=sm.grid, scale=2, maptype='satellite')
        sm = salem.Map(g.grid, factor=1, countries=False)
        ggi_img = g.get_vardata()
        sm.set_data(ggi_img)
    else:
        sm.set_topography(demp, relief_factor=.3)
    sm.set_shapefile(china_gdf, edgecolor='dimgrey', facecolor='none',
                     linewidth=3., zorder=5, ls=(0, (6, 2, 2, 2)))

    smc = copy.deepcopy(sm)
    smc.set_shapefile(rgidf, linewidth=.5,
                    facecolor='red', alpha=.5)
    smc.set_shapefile(ggidf, linewidth=.5,
                    facecolor='blue', alpha=.4)
    fig, ax = plt.subplots()
    smc.visualize(ax=ax)
    ax.scatter(x, y, marker='^', color='red', s=80,
            edgecolor='k', lw=2, zorder=150)
    ax.text(x+30, y+30, 'Peak Tomur', ha='left', va='top', fontweight='bold')
    ax.text(.3, .95, 'South Inylchek Glacier', fontweight='bold', fontsize='10',
            color='b', rotation=-0, transform=ax.transAxes,
            ha='center', va='top')
    ax.text(.6, .01, 'Tomur Glacier', fontweight='bold', fontsize='10',
            color='b', rotation=-0, transform=ax.transAxes,
            ha='left', va='bottom')
    ax.text(.85, .1, 'China', fontweight='bold', fontsize='12', color='dimgrey',
            rotation=45, transform=ax.transAxes, ha='left', va='bottom')
    ax.text(.05, .3, 'Kyrgyzstan', fontweight='bold', fontsize='12', color='dimgrey',
            rotation=45, transform=ax.transAxes, ha='left', va='bottom')
    ax.bar(0, 0, color='blue', label='GAMDAMv2', alpha=.5, edgecolor='dimgrey')
    ax.bar(0, 0, color='red', label='RGIv6', alpha=.6, edgecolor='dimgrey')
    ax.bar(0, 0, color='darkorchid', label='Both', edgecolor='dimgrey')
    ax.bar(0, 0, facecolor='none', edgecolor='dimgrey', lw=2, 
           ls=(0, (6, 2, 2, 2)), label='International\nBoundary')
    ax.legend(loc='lower left', prop=dict(weight='bold'))
    if savefig:
        if run_in_cluster:
            path = os.path.join(out_dir, 'figure')
        else:
            path = os.path.join(data_dir, 'figure')
        utils.mkdir(path)
        fig.savefig(os.path.join(path, 'glaciers_in_boundary_of_china.pdf'),
                    bbox_inches='tight', dpi=300)


thickdiff_in_largest_glc_plot(savefig=True, reset=False)
glaciers_in_boundary_of_china_plot(use_google_map=False, savefig=True)
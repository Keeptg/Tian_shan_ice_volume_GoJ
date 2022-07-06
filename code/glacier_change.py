#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:27:34 2020

@author: keeptg
"""

import os
import pandas as pd

import geopandas as gpd
from oggm import cfg, tasks, workflow, utils
from oggm.workflow import execute_entity_task
from oggm.core import gcm_climate
from calibrate_oggm_with_shean_mb import match_shean_mb
from path_config import *

def project_glacier(scenario, gcm, gi_type, dem_type, fline_type='centerlines_only',
                    cmip_type='cmip6', border=80, output_suffix='', 
                    geodetic_mb_calib='regional', dynamic_glen_a=None, glen_a_factor=5.30):
    """ run for glacier project

    Parameters
    ------
    scenario : str:
        The emission scenario for cmip.
            Should be one of in:
                ['rcp26', 'rcp45', 'rcp85'] for cmip5 
                and
                ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 
                 'ssp534-over', 'ssp585'] for cmip6
    gcm : str:
        Name of General circulation model provided by OGGM group.
            Should be one of in:
                ['CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'CanESM2', 'GFDL-CM3',
                 'GFDL-ESM2G', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR',
                 'NorESM1-M'] for cmip5
                and
                ['BCC-CSM2-MR', 'CAMS-CSM1-0', 'CESM2', 'CESM2-WACCM',
                 'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L',
                 'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR',
                 'MRI-ESM2-0', 'NorESM2-MM', 'TaiESM1'] for cmip6
    gi_type : str:
        The glacier inventory type. Should be in one of ['rgi', 'ggi']
    dem_type : str:
        The dem source used in the simulation. 
    fline_type : str:
        The type of flowline. Should be one of in ['centerlines_only', 'elev_bands']
    cmip_type : str:
        Which cmip data will be used. 
        Should be one of in ['cmip5', 'cmip6']. 
        Defualt is cmip6
    border : int
        Border of the topography.
    output_suffix : str:
        The suffix for the output data
    geodetic_mb_calib : False or str:
        If calibrate the mass balance model with Shean's (2020) geodetic mass balance model
        Should be one of in:
            [False, 'regional', 'specific']
        Default is 'regional', calibrate around the entity Tian Shan region

    """
    if cmip_type == 'cmip5':
        bp0 = 'https://cluster.klima.uni-bremen.de/~oggm/'+\
              'cmip5-ng/pr/pr_mon_{}_{}_r1i1p1_g025.nc'
        bt0 = 'https://cluster.klima.uni-bremen.de/~oggm/'+\
              'cmip5-ng/tas/tas_mon_{}_{}_r1i1p1_g025.nc'
    elif cmip_type == 'cmip6':
        bp0 = 'https://cluster.klima.uni-bremen.de/~oggm/' + \
            'cmip6/GCM/{0}/{0}_{1}_r1i1p1f1_pr.nc'
        bt0 = 'https://cluster.klima.uni-bremen.de/~oggm/' + \
            'cmip6/GCM/{0}/{0}_{1}_r1i1p1f1_tas.nc'
        bp1 = 'https://cluster.klima.uni-bremen.de/~oggm/' + \
            'cmip6/additional_GCMs/{0}/{0}_{1}_r1i1p1f1_pr.nc'
        bt1 = 'https://cluster.klima.uni-bremen.de/~oggm/' + \
            'cmip6/additional_GCMs/{0}/{0}_{1}_r1i1p1f1_tas.nc'
    else:
        raise TypeError(f"Unexcepted 'cmip_type': {cmip_type}!")

    output_path = os.path.join(out_dir, 'glacier_change_in_21st')
    working_path = os.path.join(work_dir, 'test_ice_dynamic', gi_type)
    cfg.initialize()
    cfg.PATHS['working_dir'] = utils.mkdir(working_path, reset=True)
    cfg.PARAMS['use_rgi_area'] = False
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['border'] = 80
    if dynamic_glen_a:
        cfg.PARAMS['use_inversion_params_for_run'] = False
        cfg.PARAMS['glen_a'] = dynamic_glen_a

    if gi_type == 'rgi':
        gidf = gpd.read_file(os.path.join(data_dir, 'rgi'))
    elif gi_type == 'ggi':
        gidf = gpd.read_file(os.path.join(data_dir, 'checked_gamdam'))
    else:
        raise (f"Unexcepted gi_type: {gi_type}!")

    url = 'https://cluster.klima.uni-bremen.de/~lifei/pre_process/Tianshan/{}_{}_{}'.format(gi_type,
            dem_type, fline_type)
    #    intersects_dir = utils.get_rgi_intersects_region_file(region='13')
    #    cfg.set_intersects_db(intersects_dir)

    if is_test or (not run_in_cluster):
        gidf = gidf.iloc[:5]

    gdirs = workflow.init_glacier_directories(gidf, from_prepro_level=3, prepro_border=border,
                                            prepro_base_url=url, prepro_rgi_version='60')
    
    if geodetic_mb_calib:
        match_shean_mb(gdirs, zone=geodetic_mb_calib)

    task_list = [
                 [tasks.mass_conservation_inversion, 
                  dict(glen_a=cfg.PARAMS['inversion_glen_a']*glen_a_factor)],
                  tasks.filter_inversion_output,
                  tasks.init_present_time_glacier
                 ]

    for task in task_list:
        if type(task) is list:
            execute_entity_task(task[0], gdirs, **task[1])
        else:
            execute_entity_task(task, gdirs)
    path = utils.mkdir(os.path.join(output_path, cmip_type, gi_type, output_suffix))
    utils.compile_glacier_statistics(gdirs, path=os.path.join(path, 'glacier_statistics.csv'))
    workflow.execute_entity_task(tasks.run_from_climate_data, gdirs, ys=ys, ye=2018,
                                 output_filesuffix='_spinup')
    if run_in_cluster:
        if int(os.environ.get('TASK_ID')) == 1:
            utils.compile_run_output(gdirs, input_filesuffix='_spinup',
                                     path=os.path.join(path, 'run_output'+'_history.nc'))

    # Download the files
    ft = utils.file_downloader(bt0.format(gcm, scenario))
    fp = utils.file_downloader(bp0.format(gcm, scenario))
    if not (ft and fp):
        ft = utils.file_downloader(bt1.format(gcm, scenario))
        fp = utils.file_downloader(bp1.format(gcm, scenario))
        if not(ft and fp):
            return

    run_id = '_{}_{}'.format(gcm, scenario)

    workflow.execute_entity_task(gcm_climate.process_cmip5_data, gdirs, 
                                 filesuffix='_' + run_id,  # recognize the climate file for later
                                 fpath_temp=ft,  # temperature projections
                                 fpath_precip=fp,  # precip projections
                                 year_range=('1981', '2018'),
                                 )
    workflow.execute_entity_task(tasks.run_from_climate_data, gdirs,
                                 climate_filename='gcm_data',  # use gcm_data, not climate_historical
                                 climate_input_filesuffix='_' + run_id,  # use a different scenario
                                 init_model_filesuffix='_spinup',  # this is important! Start from 2019 glacier
                                 output_filesuffix=run_id,  # recognize the run for later
                                 return_value=False,
                                 )

    # bias correct them - this overwrites the previous file
    utils.compile_run_output(gdirs, input_filesuffix=run_id, 
                             path=os.path.join(path, 'run_output' + run_id + '.nc'))

    return


intersects_dir = os.path.join(data_dir, 'gamdam_intersects')
cmip_type = 'cmip6'
is_test = False
gi_types = ['ggi', 'rgi']
dem_suffix = 'SRTM'
fline_suffix = 'centerlines_only'
calib_mb_zone = 'regional'
ys = 2000

if cmip_type == 'cmip6':
    if run_in_cluster:
        cmip6_path = '/home/www/oggm/cmip6'
    else:
        cmip6_path = data_dir
    gcms_df = pd.read_csv(os.path.join(cmip6_path, 'all_gcm_list.csv'), index_col=0)
    gcm_ssp_list = gcms_df.gcm.values + '_' + gcms_df.ssp.values
    gcms = list(set(gcms_df.gcm.to_list())) # totally 15 gcms by 02.03.2021
    gcms.sort()
    scenarios = list(set(gcms_df.ssp.to_list())) # totally 8 scenarios by 02.03.2021
    scenarios.sort()
elif cmip_type == 'cmip5':
    gcms = ['CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'CanESM2', 'GFDL-CM3',
            'GFDL-ESM2G', 'GISS-E2-R', 'IPSL-CM5A-LR', 'MPI-ESM-LR',
            'NorESM1-M']
    scenarios = ['rcp26', 'rcp45', 'rcp85']
else:
    raise ValueError(f"Unexcepted cmip_type: {cmip_type}!")


# All cmip6 run with regional geodetic mass balance calibration run
regional_geodetic_args_list = []
for scenario in scenarios:
    for gcm in gcms:
        if gcm + '_' + scenario not in gcm_ssp_list:
            continue
        for gi_type in gi_types:
            regional_geodetic_args_list.append(dict(scenario=scenario,
                                               gcm=gcm,
                                               gi_type=gi_type,
                                               dem_type='SRTM',
                                               output_suffix='SRTM'))

# All cmip6 run with specific or none geodetic and RGI
specific_none_geodetic_args_list = []
for scenario in scenarios:
    for gcm in gcms:
        if gcm + '_' + scenario not in gcm_ssp_list:
            continue
        for zone in [False]:
            specific_none_geodetic_args_list.append(dict(scenario=scenario,
                                                         gcm=gcm,
                                                         gi_type='rgi',
                                                         dem_type='SRTM',
                                                         geodetic_mb_calib=zone,
                                                         output_suffix='SRTM_{}_geodetic_calib'.format(zone if zone else 'none')))


# glen_a sensitive experiment
glen_a_sensitive_args_list = []
for scenario in scenarios:
    for gcm in gcms:
        if gcm + '_' + scenario not in gcm_ssp_list:
            continue
        for glen_a_times in [1.93, 6.18]:
            glen_a_sensitive_args_list.append(dict(scenario=scenario,
                                                   gcm=gcm,
                                                   gi_type='ggi',
                                                   dem_type='SRTM',
                                                   geodetic_mb_calib='regional',
                                                   glen_a_factor=glen_a_times, 
                                                   output_suffix=f'SRTM_({glen_a_times})'))

# dem sensitive experiment
copdem_run_args_list = []
for scenario in scenarios:
    for gcm in gcms:
        if gcm + '_' + scenario not in gcm_ssp_list:
            continue
        copdem_run_args_list.append(dict(scenario=scenario,
                                         gcm=gcm,
                                         gi_type='ggi',
                                         dem_type='COPDEM',
                                         geodetic_mb_calib='regional',
                                         output_suffix='COPDEM'))
if run_in_cluster:
    args = copdem_run_args_list[int(os.environ.get('TASK_ID'))]
else:
    args = copdem_run_args_list[-1]

project_glacier(**args)
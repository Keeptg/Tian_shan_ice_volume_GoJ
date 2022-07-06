import os
import numpy as np
from numpy.lib.type_check import _is_type_dispatcher
import pandas as pd

import netCDF4 as nc
import geopandas as gpd
import xarray as xr
import salem

from oggm import cfg, workflow, tasks, utils
from oggm.shop import gcm_climate
from oggm.workflow import execute_entity_task

from path_config import *

import matplotlib.pyplot as plt


def compiling_gcms(scenario, gcm, gi_type='ggi', dem_type='SRTM', fline_type='centerlines_only',
                    cmip_type='cmip6', border=80):
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
    cmip_type : str:
        Which cmip data will be used. 
        Should be one of in ['cmip5', 'cmip6']. 
        Defualt is cmip6
    border : int
        Border of the topography.

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
    working_path = os.path.join(work_dir, 'gcm_compiling', gi_type)
    cfg.initialize()
    cfg.PATHS['working_dir'] = utils.mkdir(working_path, reset=True)
    cfg.PARAMS['use_rgi_area'] = False
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['border'] = 80
    cfg.PARAMS['use_multiprocessing'] = True

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



    # Download the files
    ft = utils.file_downloader(bt0.format(gcm, scenario))
    fp = utils.file_downloader(bp0.format(gcm, scenario))
    if not (ft and fp):
        ft = utils.file_downloader(bt1.format(gcm, scenario))
        fp = utils.file_downloader(bp1.format(gcm, scenario))
        if not(ft and fp):
            return

    run_id = '_{}_{}'.format(gcm, scenario)

    workflow.execute_entity_task(gcm_climate.process_cmip_data, gdirs, 
                                    filesuffix='_' + run_id,  # recognize the climate file for later
                                    fpath_temp=ft,  # temperature projections
                                    fpath_precip=fp,  # precip projections
                                    year_range=('1981', '2018'),
                                    )
    output_path = os.path.join(out_dir, 'compiling_gcm')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    ds = utils.compile_climate_input(gdirs, filename='gcm_data', input_filesuffix='_'+run_id,
                                     path=os.path.join(output_path, gcm+'_'+scenario+'.nc'))


global is_test
is_test = False
gcms_list = ['BCC-CSM2-MR', 'CAMS-CSM1-0', 'CESM2', 'CESM2-WACCM',
            'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L',
            'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR',
            'MRI-ESM2-0', 'NorESM2-MM', 'TaiESM1']
scenarios_list = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

task_list = [[ssp, gcm] for gcm in gcms_list for ssp in scenarios_list]
if run_in_cluster:
    task_id = int(os.environ.get('TASK_ID'))
else:
    task_id = -1
args = task_list[task_id]
compiling_gcms(*args)
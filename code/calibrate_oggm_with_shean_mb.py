"""
This script need the newest OGGM version '1.3.2.dev120+ge9e1a0a' for now.
"""
import os
import logging
import numpy as np
import pandas as pd

import geopandas as gpd

from oggm import utils

from path_config import *

log = logging.getLogger(__name__)

def get_geodetic_mb_in_ts():
    shean_mb_path = os.path.join(data_dir, 'shean_mb.csv')
    if not os.path.exists(shean_mb_path):
        tsdf = gpd.read_file(os.path.join(data_dir, 'tienshan_region_shean', 'united_region.shp'))
        shean_mb_df = pd.read_csv(os.path.join(root_dir, 'Hma_vol', 'Input_data',
                                               'dshean_mb', 'hma_mb_20190214_1015_nmad.csv'))
        ts_rgidf = gpd.read_file(os.path.join(data_dir, 'rgi'))
        ts_rgiid = ts_rgidf.RGIId.values
        shean_mb_df['RGIId'] = ['RGI60-{:.5f}'.format(id_) for id_ in shean_mb_df.RGIId.values]
        in_tienshan = shean_mb_df.RGIId.isin(ts_rgiid)
        shean_mb_ts_df = shean_mb_df[in_tienshan]
        shean_mb_ts_df.to_csv(shean_mb_path, index=False)

    return shean_mb_path


def match_shean_mb(gdirs, zone='regional'):
    """ Calibrate the mass balance value by using the geodetic mass balance data of Shean(2020)

    Parameters
    ------
    gdirs : list of py:class:`oggm.GlacierDirectory`
    zone : str
        Calibrate the mass balance in regioanl scale or specific glacier, the specific scale
        can only be used for RGI6.0
        Should be one of ['regional', 'specific']
    """

    df = utils.compile_fixed_geometry_mass_balance(gdirs, path=False)
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    dfs = utils.compile_glacier_statistics(gdirs, path=False)
    odf = pd.DataFrame(df.loc[2000:2018].mean(), columns=['SMB'])
    odf['AREA'] = dfs.rgi_area_km2

    shean_dfs = pd.read_csv(get_geodetic_mb_in_ts())
    shean_dfs.index = shean_dfs.RGIId

    if zone == 'regional':
        smb_oggm = np.average(odf['SMB'], weights=odf['AREA'])
        smb_shean = np.average(shean_dfs['dhdt_ma'], weights=shean_dfs['area_m2']) * 1000
        residual = smb_shean - smb_oggm

        for gdir in gdirs:
            try:
                df = gdir.read_json('local_mustar')
                df['bias'] = df['bias'] - residual
                gdir.write_json(df, 'local_mustar')
            except FileNotFoundError:
                pass

        # re_compute the fixed_geometry_mass_balance to check the calibration results.
        df = utils.compile_fixed_geometry_mass_balance(gdirs, path=False)
        df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
        odf = pd.DataFrame(df.loc[2000:2018].mean(), columns=['SMB'])
        odf['AREA'] = dfs.rgi_area_km2
        smb_oggm = np.average(odf['SMB'], weights=odf['AREA'])

        bias = smb_shean - smb_oggm
        log.workflow('Shifting regional bias by {}'.format(residual))
        log.workflow('The corrected OGGM mass balance is {} mm/yr'.format(smb_oggm))
        log.workflow('The geodetic mass balance is {} mm/yr'.format(smb_shean))
        log.workflow('The bias between OGGM and geodetic mass balance is {} mm/yr'.format(bias))
    elif zone == 'specific':
        rids = []
        for gdir in gdirs:
            rid = gdir.rgi_id
            try:
                smb_oggm = odf.loc[rid]
                shean_df = shean_dfs.loc[rid]
                rids.append(rid)
            except KeyError:
                continue
            if not shean_df.empty:
                smb_shean = shean_df['dhdt_ma'] * 1000
                try:
                    residual = smb_shean - smb_oggm.SMB
                    df = gdir.read_json('local_mustar')
                    df['bias'] = df['bias'] - residual
                    gdir.write_json(df, 'local_mustar')
                except FileNotFoundError:
                    pass
        df = utils.compile_fixed_geometry_mass_balance(gdirs, path=False)
        df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
        odf = pd.DataFrame(df.loc[2000:2018].mean(), columns=['SMB'])

        shean_dfs = shean_dfs.loc[rids]
        smb_oggm_avg = np.average(odf['SMB'])
        smb_shean_avg = np.average(shean_dfs['dhdt_ma']) * 1000
        bias = smb_shean_avg - smb_oggm_avg
        log.workflow('The corrected OGGM mass balance is {} mm/yr'.format(smb_oggm_avg))
        log.workflow('The geodetic mass balance is {} mm/yr'.format(smb_shean_avg))
        log.workflow('The bias between OGGM and geodetic mass balance is {} mm/yr'.format(bias))


def test_match_shean_mb(gi_type, zone='regional'):

    from oggm import cfg, tasks, workflow
    from oggm.workflow import execute_entity_task
    if gi_type == 'rgi':
        rgidf = gpd.read_file(os.path.join(data_dir, 'rgi'))
    elif gi_type == 'ggi':
        rgidf = gpd.read_file(os.path.join(data_dir, 'checked_gamdam'))
    else:
        raise ValueError(f"Unexpected gi_type: {gi_type}!")
    if not run_in_cluster:
        rgidf = rgidf.iloc[:5]
    cfg.initialize()
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['use_rgi_area'] = False
#     cfg.PATHS['working_dir'] = utils.mkdir(os.path.join(utils.get_temp_dir(), 'test'), reset=True)
    cfg.PATHS['working_dir'] = utils.mkdir(os.path.join(work_dir, 'calibrate_oggm_with_shean_mb_test'), reset=True)
    gdirs = workflow.init_glacier_directories(rgidf)
    execute_entity_task(tasks.define_glacier_region, gdirs, source='SRTM')
    task_list = [
        tasks.glacier_masks,
        tasks.compute_centerlines,
        tasks.initialize_flowlines,
        tasks.compute_downstream_line,
        tasks.compute_downstream_bedshape,
        tasks.catchment_area,
        tasks.catchment_intersections,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction,
        tasks.process_climate_data,
        tasks.historical_climate_qc,
        tasks.local_t_star,
        tasks.mu_star_calibration,
        tasks.prepare_for_inversion,
        ]

    for task in task_list:
        workflow.execute_entity_task(task, gdirs)

    match_shean_mb(gdirs, zone=zone)


#gitypes = ['rgi', 'ggi']
#type_id = int(os.environ.get('TASK_ID'))
#type = gitypes[type_id]
#test_match_shean_mb(type)
from multiprocessing import Value
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from oggm.cli.prepro_levels import _rename_dem_folder
from oggm.cli.prepro_levels import *
from calibrate_oggm_with_shean_mb import match_shean_mb
from path_config import *


def run_prepro_levels(gi_path=None, border=None,
                      output_folder='', working_dir='', dem_source='',
                      is_test=False, test_topofile=None,
                      disable_mp=False, params_file=None, elev_bands=False,
                      match_geodetic_mb=False, centerlines_only=False,
                      add_consensus=False, max_level=3,
                      logging_level='WORKFLOW', disable_dl_verify=False):
    """Does the actual job.

    Parameters
    ----------
    gi_path : str
        the path of glacier inventory
    border : int
        the number of pixels at the maps border
    output_folder : str
        path to the output folder (where to put the preprocessed tar files)
    dem_source : str
        which DEM source to use: default, SOURCE_NAME or ALL
    working_dir : str
        path to the OGGM working directory
    params_file : str
        path to the OGGM parameter file (to override defaults)
    is_test : bool
        to test on a couple of glaciers only!
    test_topofile : str
        for testing purposes only
    test_crudir : str
        for testing purposes only
    disable_mp : bool
        disable multiprocessing
    elev_bands : bool
        compute all flowlines based on the Huss&Hock 2015 method instead
        of the OGGM default, which is a mix of elev_bands and centerlines.
    centerlines_only : bool
        compute all flowlines based on the OGGM centerline(s) method instead
        of the OGGM default, which is a mix of elev_bands and centerlines.
    match_geodetic_mb : bool
        match the regional mass-balance estimates at the regional level
        (currently Hugonnet et al., 2020).
    add_consensus : bool
        adds (reprojects) the consensus estimates thickness to the glacier
        directories. With elev_bands=True, the data will also be binned.
    max_level : int
        the maximum pre-processing level before stopping
    logging_level : str
        the logging level to use (DEBUG, INFO, WARNING, WORKFLOW)
    disable_dl_verify : bool
        disable the hash verification of OGGM downloads
    """

    # TODO: temporarily silence Fiona and other deprecation warnings
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Input check
    if max_level not in [1, 2, 3]:
        raise InvalidParamsError('max_level should be one of [1, 2, 3]')

    # Time
    start = time.time()

    def _time_log():
        # Log util
        m, s = divmod(time.time() - start, 60)
        h, m = divmod(m, 60)
        log.workflow('OGGM prepro_levels is done! Time needed: '
                     '{:02d}:{:02d}:{:02d}'.format(int(h), int(m), int(s)))

    # Config Override Params
    params = {}

    # Local paths
    utils.mkdir(working_dir)
    params['working_dir'] = working_dir

    # Initialize OGGM and set up the run parameters
    cfg.initialize(file=params_file, params=params,
                   logging_level=logging_level,
                   future=True)

    # Use multiprocessing?
    cfg.PARAMS['use_multiprocessing'] = not disable_mp

    # How many grid points around the glacier?
    # Make it large if you expect your glaciers to grow large
    cfg.PARAMS['border'] = border

    # Set to True for operational runs
    cfg.PARAMS['continue_on_error'] = True

    # Check for the integrity of the files OGGM downloads at run time
    # For large files (e.g. using a 1 tif DEM like ALASKA) calculating the hash
    # takes a long time, so deactivating this can make sense
    cfg.PARAMS['dl_verify'] = not disable_dl_verify
    cfg.PARAMS['use_rgi_area'] = False

    # Log the parameters
    msg = '# OGGM Run parameters:'
    for k, v in cfg.PARAMS.items():
        if type(v) in [pd.DataFrame, dict]:
            continue
        msg += '\n    {}: {}'.format(k, v)
    log.workflow(msg)

    rgidf = gpd.read_file(gi_path)
    split_vs_rg = rgidf.RGIId.str.split('-', expand=True)
    split_vs = split_vs_rg[0].str.split('I', expand=True)
    split_rg = split_vs_rg[1].str.split('.', expand=True)
    rgi_version = list(set(split_vs[1].to_list()))
    if len(rgi_version) > 1:
        raise ValueError(f"Got multiple rgi version number: {rgi_version}")
    else:
        rgi_version = rgi_version[0]
    rgi_reg = list(set(split_rg[0].to_list()))
    if len(rgi_reg) > 1:
        raise ValueError(f"Got multiple rgi region number: {rgi_reg}")
    else:
        rgi_reg = rgi_reg[0]
    output_base_dir = os.path.join(output_folder,
                                   'RGI{}'.format(rgi_version),
                                   'b_{:03d}'.format(border))

    # Add a package version file
    utils.mkdir(output_base_dir)
    opath = os.path.join(output_base_dir, 'package_versions.txt')
    with open(opath, 'w') as vfile:
        vfile.write(utils.show_versions(logger=log))

    if is_test:
        rgidf = rgidf.sample(5)

    log.workflow('Starting prepro run for RGI reg: {} '
                 'and border: {}'.format(rgi_reg, border))
    log.workflow('Number of glaciers: {}'.format(len(rgidf)))

    # L0 - go
    gdirs = workflow.init_glacier_directories(rgidf, reset=True, force=True)

    # Glacier stats
    sum_dir = os.path.join(output_base_dir, 'L0', 'summary')
    utils.mkdir(sum_dir)
    opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
    utils.compile_glacier_statistics(gdirs, path=opath)

    # L0 OK - compress all in output directory
    log.workflow('L0 done. Writing to tar...')
    level_base_dir = os.path.join(output_base_dir, 'L0')
    workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                 base_dir=level_base_dir)
    utils.base_dir_to_tar(level_base_dir)
    if max_level == 0:
        _time_log()
        return

    # L1 - Add dem files
    if test_topofile:
        cfg.PATHS['dem_file'] = test_topofile

    # Which DEM source?
    if dem_source.upper() == 'ALL':
        # This is the complex one, just do the job and leave
        log.workflow('Running prepro on ALL sources')
        for i, s in enumerate(utils.DEM_SOURCES):
            rs = i == 0
            log.workflow('Running prepro on sources: {}'.format(s))
            gdirs = workflow.init_glacier_directories(rgidf, reset=rs,
                                                      force=rs)
            workflow.execute_entity_task(tasks.define_glacier_region, gdirs,
                                         source=s)
            workflow.execute_entity_task(_rename_dem_folder, gdirs, source=s)

        # make a GeoTiff mask of the glacier, choose any source
        workflow.execute_entity_task(gis.rasterio_glacier_mask,
                                     gdirs, source='ALL')

        # Compress all in output directory
        level_base_dir = os.path.join(output_base_dir, 'L1')
        workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                     base_dir=level_base_dir)
        utils.base_dir_to_tar(level_base_dir)

        _time_log()
        return

    # Force a given source
    source = dem_source.upper() if dem_source else None

    # L1 - go
    workflow.execute_entity_task(tasks.define_glacier_region, gdirs,
                                 source=source)

    # Glacier stats
    sum_dir = os.path.join(output_base_dir, 'L1', 'summary')
    utils.mkdir(sum_dir)
    opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
    utils.compile_glacier_statistics(gdirs, path=opath)

    # L1 OK - compress all in output directory
    log.workflow('L1 done. Writing to tar...')
    level_base_dir = os.path.join(output_base_dir, 'L1')
    workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                 base_dir=level_base_dir)
    utils.base_dir_to_tar(level_base_dir)
    if max_level == 1:
        _time_log()
        return

    # L2 - Tasks
    # Check which glaciers will be processed as what
    if elev_bands:
        gdirs_band = gdirs
        gdirs_cent = []
    elif centerlines_only:
        gdirs_band = []
        gdirs_cent = gdirs
    else:
        # Default is to mix
        # Curated list of large (> 50 km2) glaciers that don't run
        # (CFL error) mostly because the centerlines are crap
        # This is a really temporary fix until we have some better
        # solution here
        ids_to_bands = [
            'RGI60-01.13696', 'RGI60-03.01710', 'RGI60-01.13635',
            'RGI60-01.14443', 'RGI60-03.01678', 'RGI60-03.03274',
            'RGI60-01.17566', 'RGI60-03.02849', 'RGI60-01.16201',
            'RGI60-01.14683', 'RGI60-07.01506', 'RGI60-07.01559',
            'RGI60-03.02687', 'RGI60-17.00172', 'RGI60-01.23649',
            'RGI60-09.00077', 'RGI60-03.00994', 'RGI60-01.26738',
            'RGI60-03.00283', 'RGI60-01.16121', 'RGI60-01.27108',
            'RGI60-09.00132', 'RGI60-13.43483', 'RGI60-09.00069',
            'RGI60-14.04404', 'RGI60-17.01218', 'RGI60-17.15877',
            'RGI60-13.30888', 'RGI60-17.13796', 'RGI60-17.15825',
            'RGI60-01.09783']
        if rgi_reg == '19':
            gdirs_band = gdirs
            gdirs_cent = []
        else:
            gdirs_band = []
            gdirs_cent = []
            for gdir in gdirs:
                if gdir.is_icecap or gdir.rgi_id in ids_to_bands:
                    gdirs_band.append(gdir)
                else:
                    gdirs_cent.append(gdir)

    log.workflow('Start flowline processing with: '
                 'N centerline type: {}, '
                 'N elev bands type: {}.'
                 ''.format(len(gdirs_cent), len(gdirs_band)))

    # HH2015 method
    workflow.execute_entity_task(tasks.simple_glacier_masks, gdirs_band)

    # Centerlines OGGM
    workflow.execute_entity_task(tasks.glacier_masks, gdirs_cent)

    if add_consensus:
        from oggm.shop.bedtopo import add_consensus_thickness
        workflow.execute_entity_task(add_consensus_thickness, gdirs_band)
        workflow.execute_entity_task(add_consensus_thickness, gdirs_cent)

        # Elev bands with var data
        vn = 'consensus_ice_thickness'
        workflow.execute_entity_task(tasks.elevation_band_flowline,
                                     gdirs_band, bin_variables=vn)
        workflow.execute_entity_task(tasks.fixed_dx_elevation_band_flowline,
                                     gdirs_band, bin_variables=vn)
    else:
        # HH2015 method without it
        task_list = [
            tasks.elevation_band_flowline,
            tasks.fixed_dx_elevation_band_flowline,
        ]
        for task in task_list:
            workflow.execute_entity_task(task, gdirs_band)

    # HH2015 method
    task_list = [
        tasks.compute_downstream_line,
        tasks.compute_downstream_bedshape,
    ]
    for task in task_list:
        workflow.execute_entity_task(task, gdirs_band)

    # Centerlines OGGM
    task_list = [
        tasks.compute_centerlines,
        tasks.initialize_flowlines,
        tasks.compute_downstream_line,
        tasks.compute_downstream_bedshape,
        tasks.catchment_area,
        tasks.catchment_intersections,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction,
    ]
    for task in task_list:
        workflow.execute_entity_task(task, gdirs_cent)

    # Glacier stats
    sum_dir = os.path.join(output_base_dir, 'L2', 'summary')
    utils.mkdir(sum_dir)
    opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
    utils.compile_glacier_statistics(gdirs, path=opath)

    # L2 OK - compress all in output directory
    log.workflow('L2 done. Writing to tar...')
    level_base_dir = os.path.join(output_base_dir, 'L2')
    workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                 base_dir=level_base_dir)
    utils.base_dir_to_tar(level_base_dir)
    if max_level == 2:
        _time_log()
        return

    # L3 - Tasks
    task_list = [
        tasks.process_climate_data,
        tasks.historical_climate_qc,
        tasks.local_t_star,
        tasks.mu_star_calibration,
        tasks.prepare_for_inversion,
    ]
    for task in task_list:
        workflow.execute_entity_task(task, gdirs)


    # Do we want to match geodetic estimates?
    # This affects only the bias so we can actually do this *after*
    # the inversion, but we really want to take calving into account here
    if match_geodetic_mb:
        match_shean_mb(gdirs)

    # Glacier stats
    sum_dir = os.path.join(output_base_dir, 'L3', 'summary')
    utils.mkdir(sum_dir)
    opath = os.path.join(sum_dir, 'glacier_statistics_{}.csv'.format(rgi_reg))
    utils.compile_glacier_statistics(gdirs, path=opath)
    opath = os.path.join(sum_dir, 'climate_statistics_{}.csv'.format(rgi_reg))
    utils.compile_climate_statistics(gdirs, path=opath)
    opath = os.path.join(sum_dir, 'fixed_geometry_mass_balance_{}.csv'.format(rgi_reg))
    utils.compile_fixed_geometry_mass_balance(gdirs, path=opath)

    # L3 OK - compress all in output directory
    log.workflow('L3 done. Writing to tar...')
    level_base_dir = os.path.join(output_base_dir, 'L3')
    workflow.execute_entity_task(utils.gdir_to_tar, gdirs, delete=False,
                                 base_dir=level_base_dir)
    utils.base_dir_to_tar(level_base_dir)
    if max_level == 3:
        _time_log()
        return


rgi_path = os.path.join(data_dir, 'rgi')
ggi_path = os.path.join(data_dir, 'checked_gamdam')
working_dir = os.environ.get('WORKDIR')
task_num = os.environ.get('TASK_ID')
gi_paths = [rgi_path, ggi_path]
match_geodetic_mb = False
output_folders = '/home/www/lifei/pre_process/Tianshan/{rgi}{dem}{fline}/'

gi_paths = [rgi_path, ggi_path]
border = 80
max_level = 3
run_for_test = False

args0 = dict(gi_path=rgi_path, output_folder=output_folders.format(rgi='rgi', dem='_SRTM', 
             fline='_centerlines_only'), centerlines_only=True, dem_source='SRTM')
args1 = dict(gi_path=ggi_path, output_folder=output_folders.format(rgi='ggi', dem='_SRTM', 
             fline='_centerlines_only'), centerlines_only=True, dem_source='SRTM')
args2 = dict(gi_path=rgi_path, output_folder=output_folders.format(rgi='rgi', dem='_COPDEM', 
             fline='_centerlines_only'), centerlines_only=True, dem_source='COPDEM')
args3 = dict(gi_path=ggi_path, output_folder=output_folders.format(rgi='ggi', dem='_COPDEM', 
             fline='_centerlines_only'), centerlines_only=True, dem_source='COPDEM')

args_list = [args0, args1, args2, args3]

if task_num:
    args = args_list[int(task_num)]
    run_prepro_levels(border=border, working_dir=working_dir,
                      match_geodetic_mb=match_geodetic_mb, 
                      max_level=max_level, is_test=run_for_test, 
                      **args)
else:
    pass
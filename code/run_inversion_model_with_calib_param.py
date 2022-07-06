import os
from oggm.cli.prepro_levels import run_prepro_levels
from oggm.core import flowline
from oggm.core.centerlines import compute_downstream_bedshape
import pandas as pd
import geopandas as gpd

from oggm import cfg, tasks, utils, workflow
from oggm.workflow import execute_entity_task

from Re_GlabTop import prepare_Glabtop_inputs
from Re_GlabTop import base_stress_inversion
from path_config import work_dir, out_dir, data_dir, root_dir, run_in_cluster

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


def run_inversion_model_with_calib_param(dem_type, gi_type, model_type, p_value, gidf=None, min_slope=None,
                                         suffix='', url=False, border=None,
                                         flowline_type='centerlines_only', is_test=False):
    """

    Parameters
    ----------
    dem_type : str
        The source of DEM. Should be one of ['SRTM', 'COPDEM']
    gi_type : str
        The type of glacier inventory. Should be one of ['ggi', 'rgi']
    model_type : str
        The type of inverse model. Should be one of ['oggm', 'glab']
    p_value : float
        The value of calibrated parameter.
    gidf : py:class: `geopandas.GeoDataFrame`
        The specific glacier inventory geodataframe: default is None, determined by the gi_type
    min_slope : float or int
        The value of min slope using in oggm inverse model
    bias_correct : str or False
        If the mass balance will be corrected. If yes, which kind will be used.
        Default is 'regional'.
        Should be one of ['regional', 'specific', False]
        'regional': means use the same value to shift the mass balance values for entity glaciers.
        'specific': means use the specific glacier mass balance values to calibrate the mass balance data.
            only useful when gi_type='rgi' for now.
        'none': means use the default mass balance value without any correct.
    suffix : str
        the suffix of output file name.
    url : str or bool
        the url of pre-processing myself pre-processing data.
        default is False, do not use the pre-processing data.
        if True, should be: 
            url = 'https://cluster.klima.uni-bremen.de/~lifei/pre_process/Tianshan/{}{}{}'.format(gi_type, 
                                                                                                  dem_suffix, 
                                                                                                  fline_suffix)
        or set new url with url
    border : None or int
        Set the topography border, default is None, same as cfg.PARAMS['border']
    flowline_type : str
        Which flowline will be used. Should be one of ['centerlines', 'elev_bands']
    is_test : bool
        if test function, default False

    Returns
    -------
    glc_stats : py:class:`pandas:DataFrame`
        The compiled model run output
    """


    if gi_type == 'rgi':
        if gidf is None:
            gidf = gpd.read_file(os.path.join(data_dir, 'rgi'))
    elif gi_type == 'ggi':
        if gidf is None:
            gidf = gpd.read_file(os.path.join(data_dir, 'checked_gamdam'))
    else:
        raise ValueError(f"Unexpected gi_type: {gi_type}!")

# find the pre-processing link
    if dem_type:
        dem_suffix = '_' + dem_type
    else:
        dem_suffix = ''
    if flowline_type:
        fline_suffix = '_' + flowline_type
    else: 
        fline_suffix = ''
    if url is True:
        url = 'https://cluster.klima.uni-bremen.de/~lifei/pre_process/Tianshan/{}{}{}'.format(gi_type, 
                                                                                              dem_suffix, 
                                                                                              fline_suffix)
    
    if (not run_in_cluster) or is_test:
        if len(gidf) >= 6:
            gidf = gidf.iloc[:5] 

    cfg.initialize()
    cfg.PARAMS['use_rgi_area'] = False
    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['dl_verify'] = False
    cfg.PARAMS['continue_on_error'] = True
    if border:
        cfg.PARAMS['border'] = border
    if min_slope is not None:
        cfg.PARAMS['min_slope'] = min_slope
    working_dir = os.path.join(work_dir, 'Tienshan_glc_volume_uncertainty',
                                        "{}_{}_{}_{:.2f}".format(model_type, gi_type, dem_type, p_value))
    cfg.PATHS['working_dir'] = utils.mkdir(working_dir, reset=True)

    if url:
        gdirs = workflow.init_glacier_directories(gidf, from_prepro_level=3, prepro_border=80,
                                                  prepro_base_url=url, prepro_rgi_version='60')
    else:
        gdirs = workflow.init_glacier_directories(gidf)
        execute_entity_task(tasks.define_glacier_region, gdirs, source=dem_type)
        if flowline_type == 'elev_bands':
            task_list1 = [
            tasks.simple_glacier_masks,
            tasks.elevation_band_flowline,
            tasks.fixed_dx_elevation_band_flowline,
            tasks.compute_downstream_line,
            tasks.compute_downstream_bedshape,
            ]
        elif flowline_type == 'centerlines_only':
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
            ]
        else:
            raise ValueError(f"Unexcepted flowline_type: {flowline_type}!")

        task_list2 = [
            tasks.process_climate_data,
            tasks.historical_climate_qc,
            tasks.local_t_star,
            tasks.mu_star_calibration,
            tasks.prepare_for_inversion]

        for task in task_list1+task_list2:
            execute_entity_task(task, gdirs)
    
    task_list3 = [tasks.filter_inversion_output,
                  tasks.distribute_thickness_per_altitude]
    if model_type == 'oggm':
        execute_entity_task(tasks.mass_conservation_inversion, gdirs,
                            glen_a=cfg.PARAMS['inversion_glen_a']*p_value)
    elif model_type == 'glab':
        execute_entity_task(prepare_Glabtop_inputs, gdirs)
        execute_entity_task(base_stress_inversion, gdirs, calib_tal=p_value)
    else:
        raise ValueError(f"Unexcepted model_type: {model_type}!")

    for task in task_list3[:-1]:
        execute_entity_task(task, gdirs)

    utils.mkdir(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty'))

    if suffix:
        suffix = '_' + suffix
    glc_stat = utils.compile_glacier_statistics(gdirs,
        path=os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty',
                "{}_{}_{}{}({:.2f}).csv".format(model_type, gi_type,
                                                dem_type, suffix,
                                                p_value)))

    if not run_in_cluster:
        return gdirs, glc_stat

dem_types = ['COPDEM', 'SRTM']
gi_types = ['ggi', 'rgi']
model_types = ['glab', 'oggm']
p_values_oggm = [1.93, 6.18, 5.30]
p_values_glab = [0.61, 0.71, 0.65]

# model uncertainty
args0 = dict(model_type='oggm', dem_type='SRTM', gi_type='ggi', suffix='min', p_value=p_values_oggm[0], url=True, border=80)
args1 = dict(model_type='oggm', dem_type='SRTM', gi_type='ggi', suffix='max', p_value=p_values_oggm[1], url=True, border=80)
args2 = dict(model_type='glab', dem_type='SRTM', gi_type='ggi', suffix='min', p_value=p_values_glab[0], url=True, border=80)
args3 = dict(model_type='glab', dem_type='SRTM', gi_type='ggi', suffix='max', p_value=p_values_glab[1], url=True, border=80)

# DEM uncertainty
args4 = dict(model_type='glab', dem_type='SRTM', gi_type='ggi', suffix='opt', p_value=p_values_glab[2], url=True, border=80)
args5 = dict(model_type='glab', dem_type='COPDEM', gi_type='ggi', suffix='opt', p_value=p_values_glab[2], url=True, border=80)
args6 = dict(model_type='oggm', dem_type='SRTM', gi_type='ggi', suffix='opt', p_value=p_values_oggm[2], url=True, border=80)
args7 = dict(model_type='oggm', dem_type='COPDEM', gi_type='ggi', suffix='opt', p_value=p_values_oggm[2], url=True, border=80)

# glacier inventory uncertainty
args8 = dict(model_type='glab', dem_type='SRTM', gi_type='rgi', suffix='opt', p_value=p_values_glab[2], url=True, border=80)
args9 = dict(model_type='oggm', dem_type='SRTM', gi_type='rgi', suffix='opt', p_value=p_values_oggm[2], url=True, border=80)

# failed glacier fix
args10 = dict(model_type='oggm', dem_type='COPDEM', gi_type='ggi', suffix='min', p_value=p_values_oggm[0], url=True, border=80)
args11 = dict(model_type='oggm', dem_type='COPDEM', gi_type='ggi', suffix='max', p_value=p_values_oggm[1], url=True, border=80)
args12 = dict(model_type='glab', dem_type='COPDEM', gi_type='ggi', suffix='min', p_value=p_values_glab[0], url=True, border=80)
args13 = dict(model_type='glab', dem_type='COPDEM', gi_type='ggi', suffix='max', p_value=p_values_glab[1], url=True, border=80)
args14 = dict(model_type='glab', dem_type='COPDEM', gi_type='rgi', suffix='opt', p_value=p_values_glab[2], url=True, border=80)
args15 = dict(model_type='oggm', dem_type='COPDEM', gi_type='rgi', suffix='opt', p_value=p_values_oggm[2], url=True, border=80)

# argument list
args_list = [args0, args1, args2, args3, args4, args5, args6, args7, args8,
             args9, args10, args11, args12, args13, args14, args15]

if run_in_cluster:
    args = args_list[int(os.environ.get('TASK_ID'))]
    run_inversion_model_with_calib_param(**args)
else:
    max_glc_cn = ['RGI60-13.43207']
    max_glc_ts = ['RGI60-13.05000']
    rgidf = gpd.read_file(os.path.join(data_dir, 'rgi'))
    rgidf = rgidf[rgidf.RGIId==max_glc_cn[0]]
    p_value = p_values_oggm[2]
    border = 80
    args = dict(model_type='oggm', dem_type='NASADEM', gi_type='rgi', suffix='opt', 
                p_value=p_value, url=False, 
                flowline_type='elev_bands', gidf=rgidf, border=border)
    args = args15
    gdirs, output1 = run_inversion_model_with_calib_param(**args)
    gdir = gdirs[0]
    print(output1.inv_volume_km3)

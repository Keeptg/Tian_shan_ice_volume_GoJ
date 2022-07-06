"""
This script is used to calibrate and validate the parameters of OGGM and GlabTop by using the method of
Pelto et. al (2020)

Reference
---------
Pelto, B. M., Maussion, F., Menounos, B., Radić, V., & Zeuner, M. (2020).
    Bias-corrected estimates of glacier thickness in the Columbia River Basin, Canada.
    Journal of Glaciology, 1–13. https://doi.org/10.1017/jog.2020.75
"""

import os
import multiprocessing as mp
from functools import partial
import random
from scipy.optimize import basinhopping, minimize
import numpy as np
import pandas as pd
import geopandas as gpd
from oggm import cfg, utils, workflow, tasks
from oggm.workflow import execute_entity_task

from calibrate_glabtop import get_cal_mea_diff
from Re_GlabTop import glc_inventory2oggm
from Re_GlabTop import base_stress_inversion
from Re_GlabTop import prepare_Glabtop_inputs
from Re_GlabTop import gridded_measured_thickness_point as gridded_point
from path_config import *


def to_optimize(x, gdirs, model_type, gtd=None, mae_percent=False, rewrite=False, area_weighted=True):
    """Construct the optimize function

    Parameters
    ----------
    x : list
        the calibrating parameters
    gdirs : list of py:class: `oggm.GlacierDirectory` objects
    model_type : str: oggm or glab
        the model type for calibrating
    gtd : py:class: geopandas.GeoDataFrame
        the measured ice thickness data
    mae_percent : bool
        return mean absolute error(mae) with percent or meters, default False is return in meters
    rewrite : bool
        whether rewrite the gridded measure thickness in the gridded_data.nc file,
        default False is not rewrite

    Returns
    -------
    out : float
        the mae for all calibrated glaciers
    """

    from path_config import data_dir
    task_list = [
        tasks.filter_inversion_output,
        tasks.distribute_thickness_per_altitude]
    if model_type == 'oggm':
        glen_a = cfg.PARAMS['inversion_glen_a']
        for gdir in gdirs:
            tasks.mass_conservation_inversion(gdir, glen_a=glen_a * x[0])

    elif model_type == 'glab':
        for gdir in gdirs:
            prepare_Glabtop_inputs(gdir)
            base_stress_inversion(gdir, calib_tal=x[0])
    for task in task_list:
        for gdir in gdirs:
            task(gdir)
    if gtd is None:
        gtd = gpd.read_file(os.path.join(data_dir, 'glathida2008_2014'))
    df = get_cal_mea_diff(
        gdirs, mea_thick_df=gtd, mea_thick_col='thickness', group_name=model_type, rewrite=rewrite)
    df = df.groupby('glc_name').mean()
    if area_weighted:
        me = np.average(df.thick_diff.values, weights=df.glc_area.values)
        mae = np.average(df.thick_diff.abs().values, weights=df.glc_area.values)
        mea_thick = np.average(df.mea_thick.values, weights=df.glc_area.values)
    else:
        me = np.average(df.thick_diff.values)
        mae = np.average(df.thick_diff.abs().values)
        mea_thick = np.average(df.mea_thick.values)
    if mae_percent:
        out = (me, mae, mae / mea_thick)
    else:
        out = mae
    print(f' Glacier Names: {[gdir.name for gdir in gdirs]}, Model type: {model_type}')
    print(f'x: {x}; ME_m: {me}; ME_%: {me/mea_thick}MAE_m: {mae}; '
          f'MAE_%: {mae/mea_thick}')
    return out


def splite_mea_points(gdirs, mea_point_df=None, rewrite=False, input_path=None):
    """splite the measured ice thickness dataset into trainer and tester group

    Parameters
    ----------
    gdirs : list of py:class: `oggm.GlacierDirectory` objects

    mea_point_df : py:class: `geopandas.GeoDataFrame` objects
        the measured ice thickness data
    rewrite : bool
        whether rewrite the splited thickness data, default False is not rewrite
    input_path : str
        the path to write the splited thickness data

    Returns
    -------
    trainer_path : str
        the path of trainer ice thickness dataset
    tester_path : str
        the path of tester ice thickness dataset
    """

    trainer_list = []
    tester_list = []
    if input_path is None:
        input_path = os.path.join(out_dir, 'calib_oggm_glab', 'ref_glc_validate_tienshan')
    trainer_path = os.path.join(input_path, 'trainers.shp')
    tester_path = os.path.join(input_path, 'testers.shp')
    if not os.path.exists(trainer_path) or not os.path.exists(tester_path) or rewrite:
        utils.mkdir(input_path)
        if mea_point_df is None:
            mea_point_df = gpd.read_file(os.path.join(data_dir, 'glathida2008_2014'))
        for num, gdir in enumerate(gdirs):
            outline = gdir.read_shapefile('outlines')
            outline = outline.to_crs(mea_point_df.crs.to_string())
            point_df = gpd.clip(mea_point_df, outline)
            num = len(point_df)
            num_trainer = int(num * .8)
            train_num_list = random.sample(range(num), num_trainer)
            trainer = point_df.iloc[train_num_list]
            tester = point_df.drop(point_df.index[train_num_list])
            trainer_list.append(trainer)
            tester_list.append(tester)
        trainers = pd.concat(trainer_list)
        testers = pd.concat(tester_list)
        trainers.to_file(trainer_path)
        testers.to_file(tester_path)

    return trainer_path, tester_path


def calibrate_models(gdirs, model_type, area_weighted=True):
    """Run calibrate process

    Parameters
    ----------
    gdirs : list of py:class: `oggm.GlacierDirecotry` objects
    model_type : str
        objective glacier type, should be oggm or glab
    area_weighted : bool
        whether use area_weighted mean absolute error, default is True

    """

    trainer_path, tester_path = splite_mea_points(gdirs)
    p_gdfs = gpd.read_file(trainer_path)
    t_gdfs = gpd.read_file(tester_path)
    for gdir in gdirs:
        outline = gdir.read_shapefile('outlines')
        outline = outline.to_crs(p_gdfs.crs.to_string())
        p_gdf = gpd.clip(p_gdfs, outline)
        t_gdf = gpd.clip(t_gdfs, outline)
        num = len(p_gdf)
        num_calib = int(num * .5)
        calib_num_list = random.sample(range(num), num_calib)
        calib_gdf = p_gdf.iloc[calib_num_list]
        valid_gdf = p_gdf.drop(p_gdf.index[calib_num_list])
        try:
            calib_gdfs = pd.concat([calib_gdfs, calib_gdf])
            valid_gdfs = pd.concat([valid_gdfs, valid_gdf])
            test_gdfs = pd.concat([test_gdfs, t_gdf])
        except NameError:
            calib_gdfs = calib_gdf
            valid_gdfs = valid_gdf
            test_gdfs = t_gdf
    calib_gdfs.index = range(len(calib_gdfs))
    valid_gdfs.index = range(len(valid_gdfs))
    test_gdfs.index = range(len(test_gdfs))
    for gdir in gdirs:
        gridded_point(gdir, mea_thick_df=calib_gdfs, rewrite=True, thick_col='thickness')
    calib_func1 = partial(to_optimize, gdirs=gdirs,
                          model_type=model_type, gtd=calib_gdfs, rewrite=False, area_weighted=area_weighted)
    guess, bounds = 1, ((.01, 100), )
    opti = minimize(calib_func1, x0=guess, bounds=bounds, tol=0.01)

    calib_func2 = partial(to_optimize, gdirs=gdirs, model_type=model_type,
                          gtd=valid_gdfs, mae_percent=True, rewrite=True, area_weighted=area_weighted)
    x = opti.x
    maes_calib = calib_func2(x)
    valid_func = partial(to_optimize, gdirs=gdirs, model_type=model_type, gtd=test_gdfs,
                         mae_percent=True, rewrite=True, area_weighted=area_weighted)
    maes_valid = valid_func(x)
    biases = []
    for gdir in gdirs:
        try:
            df = gdir.read_json('local_mustar')
            bias = df['bias']
            biases.append(bias)
        except FileNotFoundError:
            pass

    return [x[0], maes_calib[0], maes_calib[1], maes_calib[2], maes_valid[0],
            maes_valid[1], maes_valid[2], np.mean(biases)]


def calib_param_gdirs(gdirs, model_type='both', calib_single_glacier=False, area_weighted=True):
    """

    Parameters
    ----------
    gdirs : list of py:class: `oggm.GlacierDirectory`

    model_type : str
        the model you want to calibrate. Should be 'oggm', 'glab' or 'both'. The Default 'both' means run both
        oggm and glabtop model

    calib_single_glacier : bool
        calibrate the parameter for a single glacier or all of reference glaciers

    """

    calib_output_df = pd.DataFrame(columns=['calib_param', 'ME_calib_m', 'MAE_calib_m', 'MAE_calib_%',
                                            'ME_valid_m', 'MAE_valid_m', 'MAE_valid_%', 'mean_mb_bias', 'glc_name',
                                            'model_type'])
    if model_type == 'both':
        model_types = ['oggm', 'glab']
    else:
        model_types = [model_type]

    for model_type in model_types:
        if not calib_single_glacier:
            calib_output = calibrate_models(gdirs, model_type=model_type, area_weighted=area_weighted)
            calib_output.append('ref_glaciers')
            calib_output.append(model_type)
            index = len(calib_output_df)
            calib_output_df.loc[index] = calib_output
        else:
            for gdir in gdirs:
                name = gdir.name
                calib_output = calibrate_models([gdir], model_type=model_type, area_weighted=area_weighted)
                calib_output.append(name)
                calib_output.append(model_type)
                index = len(calib_output_df)
                calib_output_df.loc[index] = calib_output

    return calib_output_df


def calibrate(gidf, outpath=None, run_times=None, suffix='', use_shean_data=True, **kwargs):
    """Model calibrate

    Parameters
    ----------
    gidf : py:class: geopandas.GeoDataFrame
        glacier inventory dataset
    outpath : str
        the outpath for save calibration results
    run_times : int
        the iteration times for cross calibrate-validate process
    suffix : str:
        the suffix of the output file
    use_shean_data : bool
        whether or not use shean data to correct mass balance model, default is True
    **kwargs :
        the other key word argument in `calib_param_gdirs`
        model_type : one of ['oggm', 'glab', 'both']
        calib_single_glacier : True for calibrate single glacier, False for calibrate all of glaciers included in
            `gidf` dataset
    Returns
    -------

    """

    from path_config import work_dir, out_dir, data_dir
    work_dir = os.path.join(work_dir, 'ref_glc_validate_tienshan', 'working_dir')
    cfg.initialize()
    cfg.PARAMS['use_rgi_area'] = False
    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['dl_verify'] = False
    cfg.PARAMS['continue_on_error'] = False

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
        tasks.process_climate_data,
        tasks.historical_climate_qc,
        tasks.local_t_star,
        tasks.mu_star_calibration,
        tasks.prepare_for_inversion]

    gdirs_list = []
    if run_times is None:
        run_times = 100
    i = 0
    while i < run_times:
        cfg.PATHS['working_dir'] = utils.mkdir(os.path.join(work_dir, f'working_dir{i}'), reset=True)
        gdirs = workflow.init_glacier_directories(gidf)
        gdirs_list.append(gdirs)
        execute_entity_task(tasks.define_glacier_region, gdirs, source='COPDEM')

        for task in task_list1:
            execute_entity_task(task, gdirs)
        if use_shean_data:
            residual = 207.70
            for gdir in gdirs:
                try:
                    df = gdir.read_json('local_mustar')
                    df['bias'] = df['bias'] - residual
                    gdir.write_json(df, 'local_mustar')
                except FileNotFoundError:
                    pass
        i += 1

    func_gdirs = partial(calib_param_gdirs, **kwargs)
    process = mp.cpu_count()
    pool = mp.Pool(processes=process - 2)
    output_df = pd.concat(pool.map(func_gdirs, gdirs_list))
    if outpath is None:
        if suffix:
            suffix = '_' + suffix
        output_path = os.path.join(out_dir, 'calib_oggm_glab')
        outpath = os.path.join(output_path, f'calib_gdirs_global{suffix}.csv')
    if outpath:
        outpath_dir = os.path.split(outpath)[0]
        utils.mkdir(outpath_dir)
        output_df.to_csv(outpath, index=False)
    pool.close()
    pool.join()

    return output_df


def start_calibrate(**kwargs):
    """

    Parameters
    ----------
    model_type : str
        the model you want calibrated, should be one of ['oggm', 'glab', 'both']
        default is 'both'
    calib_single_glacier : bool
        True for calibrate single glacier, False for calibrate all of glaciers included in `gidf` dataset
        default is False
    run_times : int
        the iteration times for cross calibrate-validate process
    suffix : str
        the suffix for output calibrate information file
        default is ''
    use_shean_data : bool
        whether or not use shean data to correct mass balance model, default is True
    """

    ref_glc_path = os.path.join(data_dir, 'mgi2013_glathida')
    ref_glc_gdf = gpd.read_file(ref_glc_path)
    ref_glc_gdf = glc_inventory2oggm(ref_glc_gdf, assigned_col_values={'Name': ref_glc_gdf.Glc_name.values})
    output_df = calibrate(ref_glc_gdf, **kwargs)

    return output_df


def determine_p_range(**kwargs):
    """Calibrate model parameters and output the calibrate and validate informations

    Parameters
    ----------
    model_type : str
        the model you want calibrated, should be one of ['oggm', 'glab', 'both']
        default is 'both'
    run_times : int
        the iteration times for cross calibrate-validate process
    suffix : str
        the suffix for output calibrate information file
        default is ''
    use_shean_data : bool
        whether or not use shean data to correct mass balance model, default is True
    """

    from path_config import data_dir, out_dir
    ref_glc_path = os.path.join(data_dir, 'mgi2013_glathida')
    ref_glc_gdf = gpd.read_file(ref_glc_path)
    ref_glc_gdf = glc_inventory2oggm(ref_glc_gdf, assigned_col_values={'Name': ref_glc_gdf.Glc_name.values})

    p_range_df = pd.DataFrame()
    for index in range(len(ref_glc_gdf)):
        rm_gdf = ref_glc_gdf.loc[ref_glc_gdf.index[[index]]]

        gdf = ref_glc_gdf.copy()
        gdf = gdf.drop(gdf.index[[index]])
        df = calibrate(gdf, **kwargs)
        df['remove_glc'] = rm_gdf.Name.values[0]
        df_oggm = df[df.model_type == 'oggm']
        df_glab = df[df.model_type == 'glab']
        df_oggm = df_oggm[df_oggm.MAE_calib_m == df_oggm.MAE_calib_m.min()]
        df_glab = df_glab[df_glab.MAE_calib_m == df_glab.MAE_calib_m.min()]
        if len(df_oggm) > 1:
            df_oggm = df_oggm.iloc[[0]]
        if len(df_glab) > 1:
            df_glab = df_glab.iloc[[0]]
        p_range_df = pd.concat([p_range_df, df_oggm])
        p_range_df = pd.concat([p_range_df, df_glab])

    try:
        suffix = kwargs['suffix']
        p_range_df.to_csv(os.path.join(out_dir, 'calib_oggm_glab', f'p_range_{suffix}.csv'))
    except KeyError:
        p_range_df.to_csv(os.path.join(out_dir, 'calib_oggm_glab', 'p_range.csv'))


def reading_calib_result():

    import os
    import pandas as pd
    from path_config import out_dir
    pd.set_option('display.width', 320)
    pd.set_option('display.max_columns', 10)
    # cluster result
    path = os.path.join(out_dir, 'calib_oggm_glab', 'calib_gdirs_global.csv')
    calib_df = pd.read_csv(path)
    p_opt_oggm = calib_df[calib_df.model_type == 'oggm']
    p_opt_glab = calib_df[calib_df.model_type == 'glab']
    p_opt_oggm = p_opt_oggm[p_opt_oggm.MAE_valid_m == p_opt_oggm.MAE_valid_m.min()]
    p_opt_glab = p_opt_glab[p_opt_glab.MAE_valid_m == p_opt_glab.MAE_valid_m.min()]

    path = os.path.join(out_dir, 'calib_oggm_glab', 'p_range.csv')
    p_range = pd.read_csv(path)
    p_oggm = p_range[p_range.model_type == 'oggm']
    p_glab = p_range[p_range.model_type == 'glab']
    p_max_oggm = p_oggm[p_oggm.calib_param == p_oggm.calib_param.max()]
    p_min_oggm = p_oggm[p_oggm.calib_param == p_oggm.calib_param.min()]
    p_min_glab = p_glab[p_glab.calib_param == p_glab.calib_param.min()]
    p_max_glab = p_glab[p_glab.calib_param == p_glab.calib_param.max()]

    # laptop result
    path = os.path.join(os.path.join(out_dir, 'calib_oggm_glab', 'calib_gdirs.csv'))
    calib_df = pd.read_csv(path, index_col=False)

run_in_cluster = run_in_cluster
iteration_times = 100
if run_in_cluster:
    task_id = int(os.environ.get('TASK_ID'))
    args_list = [dict(model_type='oggm', suffix='with_shean_area_weighted',
                      run_times=iteration_times, use_shean_data=True, area_weighted=True),
                 dict(model_type='oggm', suffix='without_shean_area_weighted',
                      run_times=iteration_times, use_shean_data=False, area_weighted=True),
                 dict(model_type='oggm', suffix='without_shean_single_glc_area_weighted',
                      run_times=iteration_times, use_shean_data=False, calib_single_glacier=True, area_weighted=True),
                 dict(model_type='oggm', suffix='with_shean_single_glc_area_weighted', run_times=iteration_times,
                      use_shean_data=True, calib_single_glacier=True, area_weighted=True),
                 dict(model_type='oggm', suffix='without_shean_area_weighted', run_times=iteration_times,
                      use_shean_data=False, area_weighted=True),
                 dict(model_type='oggm', suffix='with_shean_area_weighted', run_times=iteration_times,
                      use_shean_data=True, area_weighted=True)]
    if task_id < 4:
        func = start_calibrate
    else:
        func = determine_p_range
    func(**args_list[task_id])
else:
    start_calibrate(model_type='oggm', suffix='with_shean', run_times=iteration_times, use_shean_data=True,
                    area_weighted=True)
#    start_calibrate(model_type='oggm', suffix='without_shean', run_times=iteration_times, use_shean_data=False)
#    start_calibrate(model_type='oggm', suffix='without_shean_single_glc', run_times=iteration_times,
#                    use_shean_data=False, calib_single_glacier=True)
#    start_calibrate(model_type='oggm', suffix='with_shean_single_glc', run_times=iteration_times,
#                    use_shean_data=True, calib_single_glacier=True)
#    determine_p_range(model_type='oggm', suffix='without_shean', run_times=iteration_times, use_shean_data=False)
#    determine_p_range(model_type='oggm', suffix='with_shean', run_times=iteration_times, use_shean_data=True)

# Data read
    calib_global_shean_df = pd.read_csv(os.path.join(cluster_dir, 'calib_oggm_glab',
                                                     'calib_gdirs_global_with_shean.csv'))
    calib_global_shean_df_opt = calib_global_shean_df[calib_global_shean_df.MAE_calib_m==
                                                    calib_global_shean_df.MAE_calib_m.min()]
    calib_global_without_shean_df = pd.read_csv(os.path.join(cluster_dir, 'calib_oggm_glab',
                                                            'calib_gdirs_global_without_shean.csv'))
    calib_global_without_shean_df_opt = calib_global_without_shean_df[calib_global_without_shean_df.MAE_calib_m==
                                                                    calib_global_without_shean_df.MAE_calib_m.min()]

    calib_global_aw_shean_df = pd.read_csv(os.path.join(cluster_dir, 'calib_oggm_glab', 
                                                        'calib_gdirs_global_with_shean_area_weighted.csv'))
    calib_global_aw_shean_df_opt = calib_global_shean_df[calib_global_aw_shean_df.MAE_calib_m==
                                                    calib_global_aw_shean_df.MAE_calib_m.min()]
    calib_global_aw_without_shean_df = pd.read_csv(os.path.join(cluster_dir, 'calib_oggm_glab',
                                                            'calib_gdirs_global_without_shean_area_weighted.csv'))
    calib_global_aw_without_shean_df_opt = calib_global_without_shean_df[calib_global_aw_without_shean_df.MAE_calib_m==
        calib_global_aw_without_shean_df.MAE_calib_m.min()]
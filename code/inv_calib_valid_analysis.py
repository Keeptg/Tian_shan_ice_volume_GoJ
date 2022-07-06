import os, re, pickle
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from oggm import cfg, utils, tasks, workflow
import oggm.core.gis

from path_config import data_dir, out_dir, work_dir, cluster_dir, root_dir
from oggm.workflow import execute_entity_task
from Re_GlabTop import prepare_Glabtop_inputs, base_stress_inversion
from Re_GlabTop import gridded_measured_thickness_point as gridded_point
from Re_GlabTop import glc_inventory2oggm

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
%matplotlib widget


def get_cal_mea_diff(gdirs, group_name, mea_thick_df=None, mea_thick_col=None, rewrite=False):
    """Extract ice thickness corresponding the input measure ice thickness point or
        'thick' attribute in 'gridded_data' dataset

    Parameter
    ---------
    gdirs : list: list of oggm.GlacierDirectory
    group_name : str: model name
    """

    mea_data_arr = np.array([])
    cal_data_arr = np.array([])
    name_data_arr = np.array([])
    for gdir in gdirs:
        if mea_thick_df is not None:
            gridded_point(gdir, mea_thick_df=mea_thick_df,
                          thick_col=mea_thick_col, rewrite=rewrite)
        gridded_data = xr.open_dataset(gdir.get_filepath('gridded_data'))
        mea_thick = gridded_data.thick.values
        cal_thick = gridded_data.distributed_thickness.values
        locate = np.where(np.isfinite(mea_thick))
        mea_data = mea_thick[locate]
        cal_data = cal_thick[locate]
        name_data = np.full(mea_data.shape, gdir.name)
        mea_data_arr = np.append(mea_data_arr, mea_data)
        cal_data_arr = np.append(cal_data_arr, cal_data)
        name_data_arr = np.append(name_data_arr, name_data)
    output = pd.DataFrame({'mea_thick': mea_data_arr,
                           'cal_thick': cal_data_arr,
                           'thick_diff': cal_data_arr - mea_data_arr,
                           'glc_name': name_data_arr,
                           'model_type': group_name})

    return output


def run_inversion_model_with_calib_param(model_type, p_value, valid_mea_data='testers'):


    gidf = gpd.read_file(os.path.join(data_dir, 'mgi2013_glathida'))
    gidf = glc_inventory2oggm(gidf, assigned_col_values={'Name': gidf.name.values})
    dem_type='COPDEM'

    cfg.initialize()
    cfg.PARAMS['use_rgi_area'] = False
    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['dl_verify'] = False
    cfg.PARAMS['continue_on_error'] = True
    working_dir = os.path.join(work_dir, 'Tienshan_glc_volume_uncertainty',
                               "{}".format(model_type))
    cfg.PATHS['working_dir'] = utils.mkdir(working_dir, reset=True)
    gdirs = workflow.init_glacier_directories(gidf)
    with open(os.path.join(cfg.PATHS['working_dir'], 'gdirs.pkl'), 'wb') as f:
        pickle.dump(gdirs, f)

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

    execute_entity_task(tasks.define_glacier_region, gdirs, source=dem_type)
    for task in task_list1:
        execute_entity_task(task, gdirs)

    if model_type == 'oggm':
        residual = 207.70
        for gdir in gdirs:
            try:
                df = gdir.read_json('local_mustar')
                df['bias'] = df['bias'] - residual
                gdir.write_json(df, 'local_mustar')
            except FileNotFoundError:
                pass

            execute_entity_task(tasks.mass_conservation_inversion, gdirs,
                                glen_a=cfg.PARAMS['inversion_glen_a']*p_value)
    elif model_type == 'glab':
        execute_entity_task(prepare_Glabtop_inputs, gdirs)
        execute_entity_task(base_stress_inversion, gdirs, calib_tal=p_value)

    for task in task_list2:
        execute_entity_task(task, gdirs)

    if valid_mea_data == 'testers':
        gtd_path = os.path.join(cluster_dir, 'tienshan_ice_volume', 'calib_oggm_glab', 'testers.shp')
    elif valid_mea_data == 'All':
        gtd_path = os.path.join(data_dir, 'glathida2008_2014')
    else:
        raise ValueError(f"Meet unexcepted 'valid_mea_data': 'f{valid_mea_data}'!")
    gtd_df = gpd.read_file(gtd_path)
    execute_entity_task(gridded_point, gdirs, mea_thick_df=gtd_df, thick_col='thickness')
    out_df = get_cal_mea_diff(gdirs, group_name=model_type)

    return out_df


def fix_faild_glc(df, df_fix, org_src, fix_src):

    f = lambda x, a: a * x**1.375
    areas = df.rgi_area_km2.values
    volumes = df.inv_volume_km3.values
    src = np.full(volumes.shape, org_src)
    fix_loc1 = np.logical_and(df.inv_volume_km3.isna(), ~df_fix.inv_volume_km3.isna())
    src = np.where(fix_loc1, fix_src, src)
    volumes = np.where(fix_loc1, df_fix.inv_volume_km3.values, volumes)
    src = np.where(np.isnan(volumes), 'VAS', src)
    xs = areas[np.isfinite(volumes)]
    ys = volumes[np.isfinite(volumes)]
    popt, _ = curve_fit(f, xs, ys)
    volumes = np.where(np.isfinite(volumes), volumes, f(areas, popt[0]))
    assert np.all(np.isfinite(volumes))
    df['inv_volume_km3'] = volumes
    df['inv_volume_src'] = src

    return df


def output_sigma_outdf():

    dir_path = os.path.join(cluster_dir, 'tienshan_ice_volume', 'Tienshan_glc_volume_uncertainty')

    if os.path.exists(os.path.join(cluster_dir, 'in_china_list.pkl')):
        with open(os.path.join(cluster_dir, 'in_china_list.pkl'), 'rb') as f:
            file = pickle.load(f)
        rgi_in_china = file['rgi_in_China']
        ggi_in_china = file['ggi_in_China']
        del file
    else:
        path = os.path.join(root_dir, 'Data', 'Shpfile', 'china-shapefiles-master',
                            'simplied_china_country.shp')

        china = gpd.read_file(path)
        ggi = gpd.read_file(os.path.join(data_dir, 'checked_gamdam'))
        rgi = gpd.read_file(os.path.join(data_dir, 'rgi'))
        rgi_in_china = [china.geometry.contains(point).values[0] for
                        point in rgi.centroid.values]
        ggi_in_china = [china.geometry.contains(point).values[0] for
                        point in ggi.centroid.values]
        with open(os.path.join(cluster_dir, 'in_china_list.pkl'), 'wb') as f:
            pickle.dump(dict(rgi_in_China=rgi_in_china,
                             ggi_in_China=ggi_in_china), f)

    # calculate result
    oggm_srtm_ggi_opt = pd.read_csv(os.path.join(dir_path, 'oggm_ggi_SRTM_opt(5.30).csv'))
    glab_srtm_ggi_opt = pd.read_csv(os.path.join(dir_path, 'glab_ggi_SRTM_opt(0.65).csv'))
    oggm_srtm_rgi_opt = pd.read_csv(os.path.join(dir_path, 'oggm_rgi_SRTM_opt(5.30).csv'))
    glab_srtm_rgi_opt = pd.read_csv(os.path.join(dir_path, 'glab_rgi_SRTM_opt(0.65).csv'))
    oggm_copdem_ggi_opt = pd.read_csv(os.path.join(dir_path, 'oggm_ggi_COPDEM_opt(5.30).csv'))
    glab_copdem_ggi_opt = pd.read_csv(os.path.join(dir_path, 'glab_ggi_COPDEM_opt(0.65).csv'))
    oggm_srtm_ggi_max = pd.read_csv(os.path.join(dir_path, 'oggm_ggi_SRTM_max(6.18).csv'))
    glab_srtm_ggi_max = pd.read_csv(os.path.join(dir_path, 'glab_ggi_SRTM_max(0.71).csv'))
    oggm_srtm_ggi_min = pd.read_csv(os.path.join(dir_path, 'oggm_ggi_SRTM_min(1.93).csv'))
    glab_srtm_ggi_min = pd.read_csv(os.path.join(dir_path, 'glab_ggi_SRTM_min(0.61).csv'))

    # fix
    oggm_copdem_ggi_max = pd.read_csv(os.path.join(dir_path, 'oggm_ggi_COPDEM_max(6.18).csv'))
    glab_copdem_ggi_max = pd.read_csv(os.path.join(dir_path, 'glab_ggi_COPDEM_max(0.71).csv'))
    oggm_copdem_ggi_min = pd.read_csv(os.path.join(dir_path, 'oggm_ggi_COPDEM_min(1.93).csv'))
    glab_copdem_ggi_min = pd.read_csv(os.path.join(dir_path, 'glab_ggi_COPDEM_min(0.61).csv'))
    oggm_copdem_rgi_opt = pd.read_csv(os.path.join(dir_path, 'oggm_rgi_COPDEM_opt(5.30).csv'))
    glab_copdem_rgi_opt = pd.read_csv(os.path.join(dir_path, 'glab_rgi_COPDEM_opt(0.65).csv'))

    oggm_srtm_ggi_opt = fix_faild_glc(oggm_srtm_ggi_opt, oggm_copdem_ggi_opt,
                                      'oggm_srtm_ggi_opt', 'oggm_COPDEM_ggi_opt')
    glab_srtm_ggi_opt = fix_faild_glc(glab_srtm_ggi_opt, glab_copdem_ggi_opt,
                                      'glab_srtm_ggi_opt', 'glab_COPDEM_ggi_opt')
    oggm_srtm_rgi_opt = fix_faild_glc(oggm_srtm_rgi_opt, oggm_copdem_rgi_opt,
                                      'oggm_srtm_rgi_opt', 'oggm_copdem_rgi_opt')
    glab_srtm_rgi_opt = fix_faild_glc(glab_srtm_rgi_opt, glab_copdem_rgi_opt,
                                      'glab_srtm_rgi_opt', 'glab_copdem_rgi_opt')
    oggm_srtm_ggi_opt.to_csv(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty', 'oggm_srtm_ggi_opt.csv'))
    glab_srtm_ggi_opt.to_csv(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty', 'glab_srtm_ggi_opt.csv'))
    oggm_srtm_rgi_opt.to_csv(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty', 'oggm_srtm_rgi_opt.csv'))
    glab_srtm_rgi_opt.to_csv(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty', 'glab_srtm_rgi_opt.csv'))

    oggm_copdem_ggi_opt = fix_faild_glc(oggm_copdem_ggi_opt, oggm_srtm_ggi_opt,
                                        'oggm_copdem_ggi_opt', 'oggm_srtm_ggi_opt')
    glab_copdem_ggi_opt = fix_faild_glc(glab_copdem_ggi_opt, glab_srtm_ggi_opt,
                                        'glab_copdem_ggi_opt', 'glab_srtm_ggi_opt')

    oggm_srtm_ggi_max = fix_faild_glc(oggm_srtm_ggi_max, oggm_copdem_ggi_max,
                                      'oggm_srtm_ggi_max', 'oggm_copdem_ggi_max')
    glab_srtm_ggi_max = fix_faild_glc(glab_srtm_ggi_max, glab_copdem_ggi_max,
                                      'glab_srtm_ggi_max', 'glab_copdem_ggi_max')
    oggm_srtm_ggi_min = fix_faild_glc(oggm_srtm_ggi_min, oggm_copdem_ggi_min,
                                      'oggm_srtm_ggi_min', 'oggm_copdem_ggi_min')
    glab_srtm_ggi_min = fix_faild_glc(glab_srtm_ggi_min, glab_copdem_ggi_min,
                                      'glab_srtm_ggi_min', 'glab_copdem_ggi_min')


    sigma_m_oggm = np.abs(oggm_srtm_ggi_max.inv_volume_km3.sum() - oggm_srtm_ggi_min.inv_volume_km3.sum())
    sigma_m_glab = np.abs(glab_srtm_ggi_max.inv_volume_km3.sum() - glab_srtm_ggi_min.inv_volume_km3.sum())
    sigma_m_oggm_ch = np.abs(oggm_srtm_ggi_max.inv_volume_km3[ggi_in_china].sum() - 
        oggm_srtm_ggi_min.inv_volume_km3[ggi_in_china].sum())
    sigma_m_glab_ch = np.abs(glab_srtm_ggi_max.inv_volume_km3[ggi_in_china].sum() - 
        glab_srtm_ggi_min.inv_volume_km3[ggi_in_china].sum())
    sigma_m_oggm_fr = np.abs(oggm_srtm_ggi_max.inv_volume_km3[ggi_in_china].sum() - 
        oggm_srtm_ggi_min.inv_volume_km3[~np.array(ggi_in_china)].sum())
    sigma_m_glab_fr = np.abs(glab_srtm_ggi_max.inv_volume_km3[ggi_in_china].sum() - 
        glab_srtm_ggi_min.inv_volume_km3[~np.array(ggi_in_china)].sum())

    sigma_dem_oggm = np.abs(oggm_srtm_ggi_opt.inv_volume_km3.sum() - oggm_copdem_ggi_opt.inv_volume_km3.sum())
    sigma_dem_glab = np.abs(glab_srtm_ggi_opt.inv_volume_km3.sum() - glab_copdem_ggi_opt.inv_volume_km3.sum())
    sigma_dem_oggm_ch = np.abs(oggm_srtm_ggi_opt.inv_volume_km3[ggi_in_china].sum() - 
        oggm_copdem_ggi_opt[ggi_in_china].inv_volume_km3.sum())
    sigma_dem_oggm_fr = np.abs(oggm_srtm_ggi_opt.inv_volume_km3[~np.array(ggi_in_china)].sum() - 
        oggm_copdem_ggi_opt.inv_volume_km3[~np.array(ggi_in_china)].sum())
    sigma_dem_glab_ch = np.abs(glab_srtm_ggi_opt.inv_volume_km3[ggi_in_china].sum() - 
        glab_copdem_ggi_opt.inv_volume_km3[ggi_in_china].sum())
    sigma_dem_glab_fr = np.abs(glab_srtm_ggi_opt.inv_volume_km3[~np.array(ggi_in_china)].sum() - 
        glab_copdem_ggi_opt.inv_volume_km3[~np.array(ggi_in_china)].sum())

    sigma_gi_oggm = np.abs(oggm_srtm_rgi_opt.inv_volume_km3.sum() - oggm_srtm_ggi_opt.inv_volume_km3.sum())
    sigma_gi_glab = np.abs(glab_srtm_rgi_opt.inv_volume_km3.sum() - glab_srtm_ggi_opt.inv_volume_km3.sum())
    sigma_gi_oggm_ch = np.abs(oggm_srtm_rgi_opt.inv_volume_km3[rgi_in_china].sum() -
        oggm_srtm_ggi_opt.inv_volume_km3[ggi_in_china].sum())
    sigma_gi_glab_ch= np.abs(glab_srtm_rgi_opt.inv_volume_km3[rgi_in_china].sum() - 
        glab_srtm_ggi_opt.inv_volume_km3[ggi_in_china].sum())
    sigma_gi_oggm_fr = np.abs(oggm_srtm_rgi_opt.inv_volume_km3[~np.array(rgi_in_china)].sum() -
        oggm_srtm_ggi_opt.inv_volume_km3[~np.array(ggi_in_china)].sum())
    sigma_gi_glab_fr= np.abs(glab_srtm_rgi_opt.inv_volume_km3[~np.array(rgi_in_china)].sum() - 
        glab_srtm_ggi_opt.inv_volume_km3[~np.array(ggi_in_china)].sum())

    sigma_v_oggm = np.sqrt(sigma_dem_oggm**2 + sigma_gi_oggm**2 + sigma_m_oggm**2)
    sigma_v_glab = np.sqrt(sigma_dem_glab**2 + sigma_gi_glab**2 + sigma_m_glab**2)
    sigma_v_oggm_ch = np.sqrt(sigma_dem_oggm_ch**2 + sigma_gi_oggm_ch**2 + sigma_m_oggm_ch**2)
    sigma_v_glab_ch = np.sqrt(sigma_dem_glab_ch**2 + sigma_gi_glab_ch**2 + sigma_m_glab_ch**2)
    sigma_v_oggm_fr = np.sqrt(sigma_dem_oggm_fr**2 + sigma_gi_oggm_fr**2 + sigma_m_oggm_fr**2)
    sigma_v_glab_fr = np.sqrt(sigma_dem_glab_fr**2 + sigma_gi_glab_fr**2 + sigma_m_glab_fr**2)

    v_oggm = oggm_srtm_ggi_opt.inv_volume_km3.sum()
    v_glab = glab_srtm_ggi_opt.inv_volume_km3.sum()
    v_oggm_ch = oggm_srtm_ggi_opt.inv_volume_km3[ggi_in_china].sum()
    v_glab_ch = glab_srtm_ggi_opt.inv_volume_km3[ggi_in_china].sum()
    v_oggm_fr = oggm_srtm_ggi_opt.inv_volume_km3[~np.array(ggi_in_china)].sum()
    v_glab_fr = glab_srtm_ggi_opt.inv_volume_km3[~np.array(ggi_in_china)].sum()

    volume_outdf = pd.DataFrame(dict(region=['Cn', 'Fr', 'Ts'],
                                     oggm=["{0:.2f} \pm {1:.2f}".format(v_oggm_ch, sigma_v_oggm_ch),
                                           "{0:.2f} \pm {1:.2f}".format(v_oggm_fr, sigma_v_oggm_fr),
                                           "{0:.2f} \pm {1:.2f}".format(v_oggm, sigma_v_oggm)],
                                     glab=["{0:.2f} \pm {1:.2f}".format(v_glab_ch, sigma_v_glab_ch),
                                           "{0:.2f} \pm {1:.2f}".format(v_glab_fr, sigma_v_glab_fr),
                                           "{0:.2f} \pm {1:.2f}".format(v_glab, sigma_v_glab)]))
    print(volume_outdf.to_latex(escape=False, index=False))

    sigma_outdf = pd.DataFrame(dict(Model=['oggm', 'glab'],
                                    sigma_m_Cn=[sigma_m_oggm_ch, sigma_m_glab_ch],
                                    sigma_m_Fr=[sigma_m_oggm_fr, sigma_m_glab_fr],
                                    sigma_dem=[sigma_dem_oggm, sigma_dem_glab],
                                    sigma_gi=[sigma_gi_oggm, sigma_gi_glab],
                                    sigma_v=[sigma_v_oggm, sigma_v_glab]))
    sigma_outdf = pd.DataFrame(dict(region=['Cn', 'Fr', 'Ts'],
        oggm_sigma_m=["{:.2f}({:.0f}%)".format(sigma_m_oggm_ch, sigma_m_oggm_ch/v_oggm_ch*1e2),
                      "{:.2f}({:.0f}%)".format(sigma_m_oggm_fr, sigma_m_oggm_fr/v_oggm_fr*1e2),
                      "{:.2f}({:.0f}%)".format(sigma_m_oggm, sigma_m_oggm/v_oggm*1e2)],
        oggm_sigma_dem=["{:.2f}({:.0f}%)".format(sigma_dem_oggm_ch, sigma_dem_oggm_ch/v_oggm_ch*1e2),
                      "{:.2f}({:.0f}%)".format(sigma_dem_oggm_fr, sigma_dem_oggm_fr/v_oggm_fr*1e2),
                      "{:.2f}({:.0f}%)".format(sigma_dem_oggm, sigma_dem_oggm/v_oggm*1e2)],
        oggm_sigma_gi=["{:.2f}({:.0f}%)".format(sigma_gi_oggm_ch, sigma_gi_oggm_ch/v_oggm_ch*1e2),
                      "{:.2f}({:.0f}%)".format(sigma_gi_oggm_fr, sigma_gi_oggm_fr/v_oggm_fr*1e2),
                      "{:.2f}({:.0f}%)".format(sigma_gi_oggm, sigma_gi_oggm/v_oggm*1e2)],
        glab_sigma_m=["{:.2f}({:.0f}%)".format(sigma_m_glab_ch, sigma_m_glab_ch/v_glab_ch*1e2),
                      "{:.2f}({:.0f}%)".format(sigma_m_glab_fr, sigma_m_glab_fr/v_glab_fr*1e2),
                      "{:.2f}({:.0f}%)".format(sigma_m_glab, sigma_m_glab/v_glab*1e2)],
        glab_sigma_dem=["{:.2f}({:.0f}%)".format(sigma_dem_glab_ch, sigma_dem_glab_ch/v_glab_ch*1e2),
                      "{:.2f}({:.0f}%)".format(sigma_dem_glab_fr, sigma_dem_glab_fr/v_glab_fr*1e2),
                      "{:.2f}({:.0f}%)".format(sigma_dem_glab, sigma_dem_glab/v_glab*1e2)],
        glab_sigma_gi=["{:.2f}({:.0f}%)".format(sigma_gi_glab_ch, sigma_gi_glab_ch/v_glab_ch*1e2),
                      "{:.2f}({:.0f}%)".format(sigma_gi_glab_fr, sigma_gi_glab_fr/v_glab_fr*1e2),
                      "{:.2f}({:.0f}%)".format(sigma_gi_glab, sigma_gi_glab/v_glab*1e2)]))
    print(sigma_outdf.to_latex(escape=False, index=False))

    return volume_outdf, sigma_outdf


def p_values_outdf():

    glc_name = ['ST', 'TYK', 'HG', 'HXLG', 'SGH', 'UMW', 'UME', 'QBT', 'ALL']
    def get_mae(cal_mea_df):
        mae = []
        me = []
        for glc in glc_name:
            if glc != 'ALL':
                df = cal_mea_df[cal_mea_df.glc_name == glc.lower().capitalize()]
            else:
                df = cal_mea_df
            mae.append((df.cal_thick-df.mea_thick).abs().mean())
            me.append((df.cal_thick-df.mea_thick).mean())
        return np.array(mae), np.array(me)

    path = os.path.join(cluster_dir, 'tienshan_ice_volume', 'calib_oggm_glab')
    calib_oggm_path = os.path.join(path, 'calib_gdirs_global_with_shean.csv')
    calib_glab_path = os.path.join(path, 'backup', 'calib_gdirs_global.csv')
    p_range_oggm_path = os.path.join(path, 'p_range_without_shean.csv')
    p_range_glab_path = os.path.join(path, 'backup', 'p_range.csv')
    calib_info_oggm = pd.read_csv(calib_oggm_path)
    calib_info_glab = pd.read_csv(calib_glab_path)
    calib_info_oggm = calib_info_oggm[calib_info_oggm.model_type=='oggm']
    calib_info_glab = calib_info_glab[calib_info_glab.model_type=='glab']
    p_oggm = calib_info_oggm[calib_info_oggm.MAE_calib_m == \
                             calib_info_oggm.MAE_calib_m.min()].calib_param.values[0]
    if type(p_oggm) is str:
        p_oggm = float(p_oggm[1:-1])
    p_glab = calib_info_glab[calib_info_glab.MAE_calib_m == \
                             calib_info_glab.MAE_calib_m.min()].calib_param.values[0]
    if type(p_glab) is str:
        p_glab = float(p_glab[1:-1])

    oggm_cal_mea_thi_p_opt = run_inversion_model_with_calib_param(model_type='oggm', p_value=p_oggm)
    glab_cal_mea_thi_p_opt = run_inversion_model_with_calib_param(model_type='glab', p_value=p_glab)
    p_opt_output = pd.concat([oggm_cal_mea_thi_p_opt, glab_cal_mea_thi_p_opt])
    p_opt_output.to_csv(os.path.join(data_dir, 'ref_glc_p_opt_mea_cal_thi.csv'))
    oggm_cal_mea_thi_p_def = run_inversion_model_with_calib_param(model_type='oggm', p_value=1)
    glab_cal_mea_thi_p_def = run_inversion_model_with_calib_param(model_type='glab', p_value=1)

    h = np.array([76.82, 55.75, 114.18, 44.37, 68.00, 65.95, 70.45, 39.33, 66.86])
    mae_f = lambda x: '{:.1f}'.format(x)
    me_f = lambda x: '{:+.1f}'.format(x)
    p_f = lambda x: '({:.0f}%)'.format(abs(x))
    oggm_p_opt_mae_m, oggm_p_opt_me_m = get_mae(oggm_cal_mea_thi_p_opt)
    oggm_p_opt_mae_p = list(map(p_f, oggm_p_opt_mae_m / h * 100))
    oggm_p_opt_mae_m = list(map(mae_f, oggm_p_opt_mae_m))
    oggm_p_opt_me_p = list(map(p_f, oggm_p_opt_me_m / h * 100))
    oggm_p_opt_me_m = list(map(me_f, oggm_p_opt_me_m))
    glab_p_opt_mae_m, glab_p_opt_me_m = get_mae(glab_cal_mea_thi_p_opt)
    glab_p_opt_mae_p = list(map(p_f, glab_p_opt_mae_m / h * 100))
    glab_p_opt_mae_m = list(map(mae_f, glab_p_opt_mae_m))
    glab_p_opt_me_p = list(map(p_f, glab_p_opt_me_m / h * 100))
    glab_p_opt_me_m = list(map(me_f, glab_p_opt_me_m))
    glab_p_def_mae_m, glab_p_def_me_m = get_mae(glab_cal_mea_thi_p_def)
    glab_p_def_mae_p = list(map(p_f, glab_p_def_mae_m / h * 100))
    glab_p_def_mae_m = list(map(mae_f, glab_p_def_mae_m))
    glab_p_def_me_p = list(map(p_f, glab_p_def_me_m / h * 100))
    glab_p_def_me_m = list(map(me_f, glab_p_def_me_m))
    oggm_p_def_mae_m, oggm_p_def_me_m = get_mae(oggm_cal_mea_thi_p_def)
    oggm_p_def_mae_p = list(map(p_f, oggm_p_def_mae_m / h * 100))
    oggm_p_def_mae_m = list(map(mae_f, oggm_p_def_mae_m))
    oggm_p_def_me_p = list(map(p_f, oggm_p_def_me_m / h * 100))
    oggm_p_def_me_m = list(map(me_f, oggm_p_def_me_m))

    out_df = pd.DataFrame({'Glacier': glc_name,
                           'H_mean': h,
                           'MC_me_def=1': np.apply_along_axis(''.join, 0, [oggm_p_def_me_m,
                                                                           oggm_p_def_me_p]),
                           'MC_mae_def=1': np.apply_along_axis(''.join, 0, [oggm_p_def_mae_m,
                                                                            oggm_p_def_mae_p]),
                           'MC_me_opt={:.2f}'.format(p_oggm): np.apply_along_axis(''.join, 0, [oggm_p_opt_me_m,
                                                                                               oggm_p_opt_me_p]),
                           'MC_mae_opt={:.2f}'.format(p_oggm): np.apply_along_axis(''.join, 0, [oggm_p_opt_mae_m,
                                                                                                oggm_p_opt_mae_p]),
                           'BS_me_def=1': np.apply_along_axis(''.join, 0, [glab_p_def_me_m,
                                                                           glab_p_def_me_p]),
                           'BS_mae_def=1': np.apply_along_axis(''.join, 0, [glab_p_def_mae_m,
                                                                            glab_p_def_mae_p]),
                           'BS_me_opt={:.2f}'.format(p_glab): np.apply_along_axis(''.join, 0, [glab_p_opt_me_m,
                                                                                               glab_p_opt_me_p]),
                           'BS_mae_opt={:.2f}'.format(p_glab): np.apply_along_axis(''.join, 0, [glab_p_opt_mae_m,
                                                                                                glab_p_opt_mae_p])})
    print(out_df.to_latex(index=False, multicolumn=20))


    return out_df


def get_my_vas(volumes, areas):

    nan_loc = volumes.isna()
    if nan_loc.sum() > 0:
        volumes = volumes.loc[~nan_loc]
        areas = areas.loc[~nan_loc]
    f = lambda x, a: a * x ** 1.375
    areas = areas.astype(float)
    volumes = volumes.astype(float)
    popt, pcov = curve_fit(f, areas, volumes)

    return popt

def status_large_glc_vol_area():

    oggm_ggi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty', 'oggm_srtm_ggi_opt.csv'))
    glab_ggi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty', 'glab_srtm_ggi_opt.csv'))
    oggm_rgi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty', 'oggm_srtm_rgi_opt.csv'))
    glab_rgi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty', 'glab_srtm_rgi_opt.csv'))

    ggi_oggm_glc_above_10 = oggm_ggi[oggm_ggi.rgi_area_km2.astype(float)>10]
    ggi_oggm_glc_under_10 = oggm_ggi[oggm_ggi.rgi_area_km2.astype(float)<=10]
    ggi_glab_glc_above_10 = glab_ggi[glab_ggi.rgi_area_km2.astype(float)>10]
    ggi_glab_glc_under_10 = glab_ggi[glab_ggi.rgi_area_km2.astype(float)<=10]
    ggi_oggm_glc_under_10.rgi_area_km2.astype(float).sum()/oggm_ggi.rgi_area_km2.astype(float).sum()
    ggi_oggm_glc_under_10.inv_volume_km3.astype(float).sum()/oggm_ggi.inv_volume_km3.astype(float).sum()
    ggi_oggm_glc_above_100 = oggm_ggi[oggm_ggi.rgi_area_km2.astype(float)>100]
    ggi_glab_glc_above_100 = glab_ggi[glab_ggi.rgi_area_km2.astype(float)>100]
    ggi_oggm_glc_above_100.inv_volume_km3.astype(float).sum()/oggm_ggi.inv_volume_km3.astype(float).sum()
    ggi_glab_glc_above_100.inv_volume_km3.astype(float).sum()/glab_ggi.inv_volume_km3.astype(float).sum()



def glc_vol_in_ch_fr(save_fig=False):

    import pickle
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (FixedLocator, MultipleLocator)
    from matplotlib.patches import Patch

    oggm_ggi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty', 'oggm_srtm_ggi_opt.csv'))
    glab_ggi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty', 'glab_srtm_ggi_opt.csv'))
    oggm_rgi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty', 'oggm_srtm_rgi_opt.csv'))
    glab_rgi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty', 'glab_srtm_rgi_opt.csv'))

    def split_df(df, area_space, by=None):

        df = df.copy()
        df['group'] = np.nan
        if not by:
            by = 'rgi_area_km2'
        for i in range(len(area_space) - 1):
            in_range = np.logical_and(df[by].astype(float) > area_space[i],
                                      df[by].astype(float) < area_space[i + 1])
            df.loc[in_range, 'group'] = area_space[i]

        return df

    def arrowed_spines(ax, spine, arrow_length=20, labels=('', ''), arrowprops=None):
        xlabel, ylabel = labels
        if ax is None:
            ax = plt.gca()
        if arrowprops is None:
            arrowprops = dict(arrowstyle='<|-', facecolor='black')

        # Set up the annotation parameters
        t = ax.spines[spine].get_transform()
        xy, xycoords = [1, 0], ('axes fraction', t)
        xytext, textcoords = [arrow_length, 0], ('offset points', t)
        if spine is 'bottom':
            ha, va = 'left', 'bottom'
        else:
            ha, va = 'right', 'top'

        # If axis is reversed, draw the arrow the other way
#        top, bottom = ax.spines[spine].axis.get_view_interval()
#        if top < bottom:
#            xy[0] = 0
#            xytext[0] *= -1
#            ha, va = 'right', 'top'

        if spine is 'bottom':
            arrow = ax.annotate(xlabel, xy, xycoords=xycoords, xytext=xytext,
                                textcoords=textcoords, ha=ha, va='center',
                                arrowprops=arrowprops)
        else:
            arrow = ax.annotate(ylabel, xy[::-1], xycoords=xycoords[::-1],
                                xytext=xytext[::-1], textcoords=textcoords[::-1],
                                ha='center', va=va, arrowprops=arrowprops)
        return arrow


    def bar_plot(ax, axt, data_list, gitype, axes_loc=False):

        area_space = [0, 1, 5, 10, 100, 500]
        models_label = ['MC', 'BS']
        gis_label = ['RGI', 'GGI']
        h_list_g = [.45, .13, .07]
        c_list_g = ['paleturquoise', 'teal']
        h_list_r = h_list_g + [.04]
        c_list_r = ['lightpink', 'r', 'maroon']
        grey_c = ['dimgrey', 'lightgrey']

        plot_loc = np.array([0, 1.2, 2.4, 3.6, 4.8])
        bar_width = h_list_g[0]
        sp = .06
        lw = 1.75
        bar_loc_g = plot_loc + (bar_width + sp) / 2
        bar_loc_r = plot_loc - (bar_width + sp) / 2

        if gitype == 'ggi':
            color = c_list_g
            height = h_list_g
            barloc = bar_loc_g
            grey_c = 'dimgrey'
        elif gitype == 'rgi':
            color = c_list_r
            height = h_list_r
            barloc = bar_loc_r
            grey_c = 'lightgrey'
        else:
            raise ValueError(f"Unexpected 'gitype': {gitype}!")

        for i, df in enumerate(data_list):
            values = df
#            a = get_my_vas(df.inv_volume_km3, df.rgi_area_km2)
            width = [values[values.group == g]['inv_volume_km3'].astype(float).sum() for
                     g in area_space[:-1]]
            nan_areas = []
            for g in area_space[:-1]:
                subdf = values[values.group == g]
                nan_area = 0
                if np.any(subdf.inv_volume_km3.isna()):
                    nan_area = subdf.iloc[np.where(subdf.inv_volume_km3.isna())].rgi_area_km2.astype(float).values
                nan_areas.append(nan_area)

            model_type = models_label[i]
            axt.barh(barloc, width, height=height[i], color=color[i],
                    alpha=.8, ec='none', lw=lw)
            if nan_areas:
                a = get_my_vas(df.inv_volume_km3, df.rgi_area_km2)
                nan_volume = [a*area**1.375 for area in nan_areas]
                nan_width = [np.sum(volume) for volume in nan_volume]
                print(f'{gitype}_{model_type}:')
                print(f"Failed cases number: {np.sum([len(data) for data in nan_areas if data is not 0])}")
                print(f"Failed cases Areas: {np.sum([np.sum([area]) for area in nan_areas])}")
                print(f"Failed case volumes: {np.sum(nan_width)}")
    #            print(f"Calibrated c: {a}")
                print(f"Totle glacier area: {df.rgi_area_km2.astype(float).sum()}")
                axt.barh(barloc, nan_width, height=height[i], ec=color[i],
                        alpha=.8, fc='none', lw=lw, left=width)
            length = [len(values[values.group == g]) for
                      g in area_space[:-1]]
            areas = [values[values.group == g].rgi_area_km2.astype(float).sum() for
                     g in area_space[:-1]]
            bar_g = ax.barh(barloc, areas, height=height[0], color=grey_c,
                             lw=lw)
            text_w = np.array([bar.get_width() for
                               bar in bar_g.get_children()]) * 1.01
            for j, l in enumerate(length):
                ax.text(text_w[j], barloc[j], "{} ".format(int(l)),
                         ha='left', va='center', fontsize=8, fontweight='bold')

        ax.set_yticks(plot_loc)
        ax.set_yticklabels(['0~1',
                            '>5',
                            '>10',
                            '>100',
                            '>500'],
                           rotation=90, va='center')
        ylim = ax.get_ylim()
#       axt.set_xscale('log')
        ax.set_xlim(1., 4000)
        axt.set_facecolor('none')
        axt.set_zorder(1)
        axt.set_xlim(0, 400)
        arrow_loc = []
        if axes_loc:
            axt.invert_xaxis()
            if axes_loc == 'top':
                axt.spines['bottom'].set_color('lightgrey')
                axt.spines['top'].set_color('none')
                ax.spines['top'].set_color('none')
                ax.spines['bottom'].set_color('lightgrey')
                arrow_loc.append(ylim[1])
                axt.arrow(x=0, y=ylim[1], dx=175, dy=0, head_width=.3,
                          head_length=10, color='k')
                ax.tick_params(axis='x', top=True, bottom=False,
                               labeltop=True, labelbottom=False, which='both')
                ax.xaxis.set_minor_locator(FixedLocator(range(0, 1701, 100)))
                ax.arrow(x=0, y=ylim[1], dx=1800, dy=0, head_width=.3, head_length=80,
                          color='k')
                ax.set_xticks(range(0, 1800, 500))
                ax.set_xlabel('Glacier Area (km$^2$)', va='center')
                ax.xaxis.set_label_coords(0.25, 1.17, transform=ax.transAxes)
                axt.set_xticks([0, 50, 100, 150])
                axt.set_xlabel('Ice volume (km$^3$)', va='center')
                axt.xaxis.set_label_coords(0.8, 1.17, transform=ax.transAxes)
                axt.xaxis.set_minor_locator(FixedLocator(range(0, 171, 10)))
                axt.set_ylabel('Glaciers in China')
                #axt.arrow(x=0, y=ylim[1], dx=1800, dy=0, head_width=.3, head_length=80,
                #          color='k')
                #axt.set_xticks(range(0, 1800, 500))
                #axt.xaxis.set_minor_locator(FixedLocator(range(0, 1701, 100)))
                #axt.set_xlabel('Glacier Area ($\mathregular{km^2}$)')
                #axt.xaxis.set_label_coords(0.68, 1.17)
            elif axes_loc == 'bottom':
                axt.spines['top'].set_color('lightgrey')
                axt.spines['bottom'].set_color('none')
                ax.spines['bottom'].set_color('k')
                arrow_loc.append(ylim[0])
                ax.set_ylabel('Glacier area range (km$^2$)')
                ax.yaxis.set_label_coords(-.07, 1)
                ax.tick_params(axis='x', which='both',
                               top=False, labeltop=False, bottom=False, labelbottom=False)

                #ax.set_xticks([0, 50, 100, 150])
                #ax.set_ylabel('Glacier Area Range ($\mathregular{km^2}$)')
                #ax.yaxis.set_label_coords(-.07, 1)
                #ax.xaxis.set_minor_locator(FixedLocator(range(0, 171, 10)))
                #axt.minorticks_off()
                axt.set_xticks([])
                axt.tick_params(axis='x', colors='dimgrey')
                ax.spines['top'].set_color('lightgrey')

        return ax, axt, arrow_loc

    from path_config import root_dir
    try:
        path = os.path.join(root_dir, 'Data', 'Shpfile', 'china-shapefiles-master',
                            'simplied_china_country.shp')
        china = gpd.read_file(path)
    except:
        path = os.path.join(data_dir, 'china-shapefiles-master',
                            'simplied_china_country.shp')
        china = gpd.read_file(path)
    if os.path.exists(os.path.join(cluster_dir, 'in_china_list.pkl')):
        with open(os.path.join(cluster_dir, 'in_china_list.pkl'), 'rb') as f:
            file = pickle.load(f)
        rgi_in_china = file['rgi_in_China']
        gam_in_china = file['ggi_in_China']
        del file
    else:
        ggi = gpd.read_file(os.path.join(data_dir, 'checked_gamdam'))
        rgi = gpd.read_file(os.path.join(data_dir, 'rgi'))
        rgi_in_china = [china.geometry.contains(point).values[0] for
                        point in rgi.centroid.values]
        gam_in_china = [china.geometry.contains(point).values[0] for
                        point in ggi.centroid.values]
        with open(os.path.join(cluster_dir, 'in_china_list.pkl'), 'wb') as f:
            pickle.dump(dict(rgi_in_China=rgi_in_china,
                             ggi_in_China=gam_in_china), f)

    oggm_rgi_china_vol = oggm_rgi.loc[rgi_in_china]
    glab_rgi_china_vol = glab_rgi.loc[rgi_in_china]
    oggm_rgi_forign_vol = oggm_rgi.loc[~np.array(rgi_in_china)]
    glab_rgi_forign_vol = glab_rgi.loc[~np.array(rgi_in_china)]
    oggm_gam_china_vol = oggm_ggi.loc[gam_in_china]
    glab_gam_china_vol = glab_ggi.loc[gam_in_china]
    oggm_gam_forign_vol = oggm_ggi.loc[~np.array(gam_in_china)]
    glab_gam_forign_vol = glab_ggi.loc[~np.array(gam_in_china)]

    area_space = [0, 1, 5, 10, 100, 500]

#    models_label = ['MB', 'BS']
#    gis_label = ['RGI', 'GGI']
    h_list_g = [.45, .13, .07]
    c_list_g = ['paleturquoise', 'teal']
#    h_list_r = h_list_g + [.04]
    c_list_r = ['lightpink', 'r', 'maroon']  # mistyrose, 'lightcoral',

    plot_loc = np.array([0, 1.2, 2.4, 3.6, 4.8])
    bar_width = h_list_g[0]
    sp = .06
#    lw = 1.75
#    bar_loc_g = plot_loc + (bar_width + sp) / 2
#    bar_loc_r = plot_loc - (bar_width + sp) / 2

    splited_rgidf_ch_oggm = split_df(oggm_rgi_china_vol, area_space,
                                     'rgi_area_km2')
    splited_ggidf_ch_oggm = split_df(oggm_gam_china_vol, area_space,
                                     'rgi_area_km2')
    splited_rgidf_fr_oggm = split_df(oggm_rgi_forign_vol, area_space,
                                     'rgi_area_km2')
    splited_ggidf_fr_oggm = split_df(oggm_gam_forign_vol, area_space,
                                     'rgi_area_km2')
    splited_rgidf_ch_glab = split_df(glab_rgi_china_vol, area_space,
                                     'rgi_area_km2')
    splited_ggidf_ch_glab = split_df(glab_gam_china_vol, area_space,
                                     'rgi_area_km2')
    splited_rgidf_fr_glab = split_df(glab_rgi_forign_vol, area_space,
                                     'rgi_area_km2')
    splited_ggidf_fr_glab = split_df(glab_gam_forign_vol, area_space,
                                     'rgi_area_km2')

    fig, axs = plt.subplots(2, 1, figsize=(5.5, 6))
    plt.subplots_adjust(hspace=0, bottom=.2)
    ax1 = axs[0]
    axt1 = ax1.twiny()
    print("\n")
    print("In China")
    bar_plot(ax=ax1, axt=axt1, data_list=[splited_ggidf_ch_oggm, splited_ggidf_ch_glab],
             gitype='ggi')
    print("\n")
    print("In China")
    bar_plot(ax=ax1, axt=axt1, data_list=[splited_rgidf_ch_oggm, splited_rgidf_ch_glab],
             gitype='rgi', axes_loc='top')

    ax2 = axs[1]
    axt2 = ax2.twiny()
    print("\n")
    print('Out China')
    bar_plot(ax=ax2, axt=axt2, data_list=[splited_ggidf_fr_oggm, splited_ggidf_fr_glab],
             gitype='ggi')
    print("\n")
    print('Out China')
    bar_plot(ax=ax2, axt=axt2, data_list=[splited_rgidf_fr_oggm, splited_rgidf_fr_glab],
             gitype='rgi', axes_loc='bottom')

    label_dict1 = {}
    label_dict2 = {}
    for color, lb in zip([c_list_g[0], c_list_r[0], c_list_g[1], c_list_r[1]],
                         ['MC/GAMDAMv2', 'MC/RGIv6', 'BS/GAMDAMv2', 'BS/RGIv6']):
        label_dict1[lb] = Patch(color=color, label=lb)
    for color, lb in zip(['dimgrey', 'lightgrey'], ['GAMDAMv2', 'RGIv6']):
        label_dict2[lb] = Patch(color=color, label=lb)
    ax2.legend(handles=[label_dict2['GAMDAMv2'], label_dict2['RGIv6']],
               bbox_to_anchor=(0.01, .07), loc='upper left', ncol=1,
               facecolor='w', framealpha=1, labelspacing=.1, borderpad=.2)

#    ax1.add_artist(l1)

#    l2 = ax2.legend(m2, ['GGI', 'RGI'], handlelength=1, handletextpad=.1, columnspacing=.7,
#                    bbox_to_anchor=(1.00, 0), loc='upper right', ncol=2)
    axt2.legend(handles=[label_dict1['MC/GAMDAMv2'], label_dict1['BS/GAMDAMv2'],
                         label_dict1['MC/RGIv6'], label_dict1['BS/RGIv6']],
                bbox_to_anchor=(0.99, .07), loc='upper right', ncol=2,
                facecolor='w', framealpha=1, labelspacing=.1, columnspacing=.4,
                borderpad=.2)
    ax1.set_ylim(-0.768, 5.568)
    ax2.set_ylim(-0.767, 5.568)
    ax1.text(1.01, .5, 'In China', transform=ax1.transAxes, ha='left',
             va='center', weight='bold', rotation=90)
    ax2.text(1.01, .5, 'Outside China', transform=ax2.transAxes, ha='left',
             va='center', weight='bold', rotation=90)

    if save_fig:
        fig.savefig(os.path.join(data_dir, 'figure',
                                 'Ice_vol_from_MC_BS_RGI_GGI.pdf'),
                    bbox_inches='tight', dpi=300)

    # glc_vol_in_ch_fr()

    nan_rgi = oggm_rgi.loc[oggm_rgi.inv_volume_km3.isna()]
    nan_ggi = oggm_ggi.loc[oggm_ggi.inv_volume_km3.isna()]
    rgi_error_msg = nan_rgi.error_msg
    ggi_error_msg = nan_ggi.error_msg

    return fig, ax1, axt1, ax2, axt2


def main():

    axs = glc_vol_in_ch_fr(save_fig=True)

    output_sigma_outdf()

    import pickle
    oggm_ggi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty',
                                          'oggm_srtm_ggi_opt.csv'))
    glab_ggi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty',
                                          'glab_srtm_ggi_opt.csv'))
    oggm_rgi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty',
                                          'oggm_srtm_rgi_opt.csv'))
    glab_rgi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty',
                                          'glab_srtm_rgi_opt.csv'))

    with open(os.path.join(cluster_dir, 'in_china_list.pkl'), 'rb') as f:
        file = pickle.load(f)
    rgi_in_china = file['rgi_in_China']
    gam_in_china = file['ggi_in_China']
    del file
    oggm_ggi['in_china'] = gam_in_china
    oggm_rgi['in_china'] = rgi_in_china
    glab_ggi['in_china'] = gam_in_china
    glab_rgi['in_china'] = rgi_in_china

    oggm_ggi_ch = oggm_ggi[oggm_ggi.in_china]
    oggm_ggi_fr = oggm_ggi[~oggm_ggi.in_china]
    oggm_rgi_ch = oggm_rgi[oggm_rgi.in_china]
    oggm_rgi_fr = oggm_rgi[~oggm_rgi.in_china]
    oggm_ggi_10_ch = oggm_ggi_ch[oggm_ggi_ch.rgi_area_km2.astype(float) > 10]
    oggm_rgi_10_ch = oggm_rgi_ch[oggm_rgi_ch.rgi_area_km2.astype(float) > 10]
    oggm_ggi_10_fr = oggm_ggi_fr[oggm_ggi_fr.rgi_area_km2.astype(float) > 10]
    oggm_rgi_10_fr = oggm_rgi_fr[oggm_rgi_fr.rgi_area_km2.astype(float) > 10]

    oggm_rgi_10_ch_vol = oggm_rgi_10_ch.inv_volume_km3.astype(float).sum()
    oggm_rgi_10_fr_vol = oggm_rgi_10_fr.inv_volume_km3.astype(float).sum()
    oggm_ggi_10_ch_vol = oggm_ggi_10_ch.inv_volume_km3.astype(float).sum()
    oggm_ggi_10_fr_vol = oggm_ggi_10_fr.inv_volume_km3.astype(float).sum()

    oggm_rgi_ggi_ch_diff = oggm_rgi_ch.rgi_area_km2.astype(float).sum() -\
        oggm_ggi_ch.rgi_area_km2.astype(float).sum()
    oggm_rgi_ggi_ch_10_diff = oggm_rgi_10_ch.rgi_area_km2.astype(float).sum() - \
        oggm_ggi_10_ch.rgi_area_km2.astype(float).sum()
    oggm_ggi_ch_area = oggm_ggi_ch.rgi_area_km2.astype(float).sum()
    oggm_rgi_ch_area = oggm_rgi_ch.rgi_area_km2.astype(float).sum()
    oggm_ggi_ch_vol = oggm_ggi_ch.inv_volume_km3.astype(float).sum()
    oggm_rgi_ch_vol = oggm_rgi_ch.inv_volume_km3.astype(float).sum()

    oggm_rgi_ggi_fr_diff = oggm_rgi_fr.rgi_area_km2.astype(float).sum() -\
        oggm_ggi_fr.rgi_area_km2.astype(float).sum()
    oggm_rgi_ggi_fr_10_diff = oggm_rgi_10_fr.rgi_area_km2.astype(float).sum() -\
         oggm_ggi_10_fr.rgi_area_km2.astype(float).sum()
    oggm_ggi_fr_area = oggm_ggi_fr.rgi_area_km2.astype(float).sum()
    oggm_rgi_fr_area = oggm_rgi_fr.rgi_area_km2.astype(float).sum()
    oggm_ggi_fr_vol = oggm_ggi_fr.inv_volume_km3.astype(float).sum()
    oggm_rgi_fr_vol = oggm_rgi_fr.inv_volume_km3.astype(float).sum()

    glab_ggi_ch = glab_ggi[glab_ggi.in_china]
    glab_ggi_fr = glab_ggi[~glab_ggi.in_china]
    glab_rgi_ch = glab_rgi[glab_rgi.in_china]
    glab_rgi_fr = glab_rgi[~glab_rgi.in_china]
    glab_ggi_10_ch = glab_ggi_ch[glab_ggi_ch.rgi_area_km2.astype(float) > 10]
    glab_rgi_10_ch = glab_rgi_ch[glab_rgi_ch.rgi_area_km2.astype(float) > 10]
    glab_ggi_10_fr = glab_ggi_fr[glab_ggi_fr.rgi_area_km2.astype(float) > 10]
    glab_rgi_10_fr = glab_rgi_fr[glab_rgi_fr.rgi_area_km2.astype(float) > 10]

    glab_rgi_10_ch_vol = glab_rgi_10_ch.inv_volume_km3.astype(float).sum()
    glab_rgi_10_fr_vol = glab_rgi_10_fr.inv_volume_km3.astype(float).sum()
    glab_ggi_10_ch_vol = glab_ggi_10_ch.inv_volume_km3.astype(float).sum()
    glab_ggi_10_fr_vol = glab_ggi_10_fr.inv_volume_km3.astype(float).sum()

    glab_rgi_ggi_ch_diff = glab_rgi_ch.rgi_area_km2.astype(float).sum() - \
        glab_ggi_ch.rgi_area_km2.astype(float).sum()
    glab_rgi_ggi_ch_10_diff = glab_rgi_10_ch.rgi_area_km2.astype(float).sum() -\
         glab_ggi_10_ch.rgi_area_km2.astype(float).sum()
    glab_ggi_ch_area = glab_ggi_ch.rgi_area_km2.astype(float).sum()
    glab_rgi_ch_area = glab_rgi_ch.rgi_area_km2.astype(float).sum()
    glab_ggi_ch_vol = glab_ggi_ch.inv_volume_km3.astype(float).sum()
    glab_rgi_ch_vol = glab_rgi_ch.inv_volume_km3.astype(float).sum()

    glab_rgi_ggi_fr_diff = glab_rgi_fr.rgi_area_km2.astype(float).sum() -\
         glab_ggi_fr.rgi_area_km2.astype(float).sum()
    glab_rgi_ggi_fr_10_diff = glab_rgi_10_fr.rgi_area_km2.astype(float).sum() -\
         glab_ggi_10_fr.rgi_area_km2.astype(float).sum()
    glab_ggi_fr_area = glab_ggi_fr.rgi_area_km2.astype(float).sum()
    glab_rgi_fr_area = glab_rgi_fr.rgi_area_km2.astype(float).sum()
    glab_ggi_fr_vol = glab_ggi_fr.inv_volume_km3.astype(float).sum()
    glab_rgi_fr_vol = glab_rgi_fr.inv_volume_km3.astype(float).sum()


def add_all_glc_vol_area_plot(savefig=False):
    oggm_ggi = gpd.read_file(os.path.join(out_dir,
                                        'Tienshan_glc_volume_uncertainty',
                                        'oggm_srtm_ggi_opt.csv'))
    glab_ggi = gpd.read_file(os.path.join(out_dir,
                                        'Tienshan_glc_volume_uncertainty',
                                        'glab_srtm_ggi_opt.csv'))
    oggm_rgi = gpd.read_file(os.path.join(out_dir,
                                        'Tienshan_glc_volume_uncertainty',
                                        'oggm_srtm_rgi_opt.csv'))
    glab_rgi = gpd.read_file(os.path.join(out_dir,
                                        'Tienshan_glc_volume_uncertainty',
                                        'glab_srtm_rgi_opt.csv'))
    rgi_area = oggm_rgi.rgi_area_km2.astype(float).sum()
    ggi_area = oggm_ggi.rgi_area_km2.astype(float).sum()
    oggm_rgi_vol = oggm_rgi.inv_volume_km3.astype(float).sum()
    oggm_ggi_vol = oggm_ggi.inv_volume_km3.astype(float).sum()
    glab_ggi_vol = glab_ggi.inv_volume_km3.astype(float).sum()
    glab_rgi_vol = glab_rgi.inv_volume_km3.astype(float).sum()

    fig, ax1, axt1, ax2, axt2 = glc_vol_in_ch_fr(save_fig=False)
    bbox = ax2.get_position()
    ax3 = fig.add_axes([bbox.x0, bbox.y0-.13, bbox.x1-bbox.x0, .1])
    ax3.set_zorder(-2)
    ax3.tick_params(axis='y', left=False, labelleft=False)
    ax3.set_ylim(0, 2)
    ax3.barh(1.3, ggi_area, .55, fc='dimgrey', align='center')
    ax3.barh(.7, rgi_area, .55, fc='lightgrey', align='center')
    ax3.set_xlim(0, 26000)
    ax3.spines['bottom'].set_color('none')
    ax3.arrow(0, 0, 12500, 0, color='k', head_width=.3, head_length=800)
    ax3.xaxis.set_minor_locator(FixedLocator(range(0, 12500, 1000)))
    ax3.set_xticks([0, 5000, 10000])
    ax3.text(1.01, .5, 'All', transform=ax3.transAxes, ha='left',
                va='center', weight='bold', rotation=90)
    ax3.text(ggi_area, 1.3, len(oggm_ggi), fontweight='bold', fontsize=8, va='center', ha='left')
    ax3.text(rgi_area, .7, len(oggm_rgi), fontweight='bold', fontsize=8, va='center', ha='left')

    c_list_g = ['paleturquoise', 'teal']
    c_list_r = ['lightpink', 'r', 'maroon']
    axt3 = ax3.twiny()
    axt3.set_zorder(-1)
    axt3.spines['bottom'].set_color('none')
    axt3.barh(1.3, oggm_ggi_vol, .55, fc=c_list_g[0], align='center')
    axt3.barh(.7, oggm_rgi_vol, .55, fc=c_list_r[0], align='center')
    axt3.barh(1.3, glab_ggi_vol, .2, fc=c_list_g[1], align='center')
    axt3.barh(.7, glab_rgi_vol, .2, fc=c_list_r[1], align='center')
    axt3.set_xlim(0, 1700)
    axt3.invert_xaxis()
    axt3.tick_params(axis='x', which='both', top=False, labeltop=False, bottom=True, labelbottom=True)
    axt3.set_xticks(range(0, 700, 200))
    axt3.xaxis.set_minor_locator(FixedLocator(range(0, 700, 40)))
    axt3.arrow(0, 0, 710, 0, head_width=.3, head_length=60, color='k')
    ax3.set_xlabel('Glacier area (km$^2$)', va='center')
    ax3.xaxis.set_label_coords(.25, -.55)
    axt3.set_xlabel('Ice volume (km$^3$)', va='center')
    axt3.xaxis.set_label_coords(.8, -.55)
    if savefig:
        fig.savefig(os.path.join(data_dir, 'figure',
                                    'Ice_vol_from_MC_BS_RGI_GGI.pdf'),
                    bbox_inches='tight', dpi=300)


def compared_vas_inv_model():
    def find_float(x):
        try:
            float(x)
            return True
        except:
            return False
    oggm_ggi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty',
                                          'oggm_srtm_ggi_opt.csv'))
    glab_ggi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty',
                                          'glab_srtm_ggi_opt.csv'))
    oggm_rgi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty',
                                          'oggm_srtm_rgi_opt.csv'))
    glab_rgi = gpd.read_file(os.path.join(out_dir, 'Tienshan_glc_volume_uncertainty',
                                          'glab_srtm_rgi_opt.csv'))
    oggm_ggi_checked = oggm_ggi[oggm_ggi.vas_volume_km3.apply(find_float)]
    oggm_ggi_checked['vas_bahr'] = 0.034 * oggm_ggi_checked.rgi_area_km2.astype(float)**\
        1.375
    oggm_ggi_checked['vas_grinsted'] = 0.043 * oggm_ggi_checked.rgi_area_km2.astype(float)**\
        1.29
    oggm_ggi_checked.vas_grinsted.sum()
    oggm_ggi1km2 = oggm_ggi_checked[oggm_ggi_checked.rgi_area_km2.astype(float)<1]
    (oggm_ggi1km2.vas_bahr.sum() - oggm_ggi1km2.inv_volume_km3.astype(float).sum())/\
        oggm_ggi1km2.inv_volume_km3.astype(float).sum()
    
    (oggm_ggi1km2.vas_grinsted.sum() - oggm_ggi1km2.inv_volume_km3.astype(float).sum())/\
        oggm_ggi1km2.inv_volume_km3.astype(float).sum()

    oggm_ggi10km2 = oggm_ggi_checked[(oggm_ggi_checked.rgi_area_km2.astype(float)>1) &
                                     (oggm_ggi_checked.rgi_area_km2.astype(float)<10)]
    (oggm_ggi10km2.vas_bahr.sum() - oggm_ggi10km2.inv_volume_km3.astype(float).sum())/\
        oggm_ggi10km2.inv_volume_km3.astype(float).sum()
    
    (oggm_ggi10km2.vas_grinsted.sum() - oggm_ggi10km2.inv_volume_km3.astype(float).sum())/\
        oggm_ggi10km2.inv_volume_km3.astype(float).sum()
    
    oggm_ggi100km2 = oggm_ggi_checked[oggm_ggi_checked.rgi_area_km2.astype(float)>10]
    (oggm_ggi100km2.vas_bahr.sum() - oggm_ggi100km2.inv_volume_km3.astype(float).sum())/\
        oggm_ggi100km2.inv_volume_km3.astype(float).sum()
    
    (oggm_ggi100km2.vas_grinsted.sum() - oggm_ggi100km2.inv_volume_km3.astype(float).sum())/\
        oggm_ggi100km2.inv_volume_km3.astype(float).sum()

    glab_ggi_checked = glab_ggi[glab_ggi.vas_volume_km3.apply(find_float)]
    glab_ggi_checked['vas_bahr'] = 0.034 * glab_ggi_checked.rgi_area_km2.astype(float)**\
        1.375
    glab_ggi_checked['vas_grinsted'] = 0.043 * glab_ggi_checked.rgi_area_km2.astype(float)**\
        1.29
    glab_ggi_checked.vas_grinsted.sum()
    glab_ggi1km2 = glab_ggi_checked[glab_ggi_checked.rgi_area_km2.astype(float)<1]
    (glab_ggi1km2.vas_bahr.sum() - glab_ggi1km2.inv_volume_km3.astype(float).sum())/\
        glab_ggi1km2.inv_volume_km3.astype(float).sum()
    
    (glab_ggi1km2.vas_grinsted.sum() - glab_ggi1km2.inv_volume_km3.astype(float).sum())/\
        glab_ggi1km2.inv_volume_km3.astype(float).sum()

    glab_ggi10km2 = glab_ggi_checked[(glab_ggi_checked.rgi_area_km2.astype(float)>1) &
                                     (glab_ggi_checked.rgi_area_km2.astype(float)<10)]
    (glab_ggi10km2.vas_bahr.sum() - glab_ggi10km2.inv_volume_km3.astype(float).sum())/\
        glab_ggi10km2.inv_volume_km3.astype(float).sum()
    
    (glab_ggi10km2.vas_grinsted.sum() - glab_ggi10km2.inv_volume_km3.astype(float).sum())/\
        glab_ggi10km2.inv_volume_km3.astype(float).sum()
    
    glab_ggi100km2 = glab_ggi_checked[glab_ggi_checked.rgi_area_km2.astype(float)>10]
    (glab_ggi100km2.vas_bahr.sum() - glab_ggi100km2.inv_volume_km3.astype(float).sum())/\
        glab_ggi100km2.inv_volume_km3.astype(float).sum()
    
    (glab_ggi100km2.vas_grinsted.sum() - glab_ggi100km2.inv_volume_km3.astype(float).sum())/\
        glab_ggi100km2.inv_volume_km3.astype(float).sum()
    return
# p_values_outdf()
# output_sigma_outdf()
# glc_vol_in_ch_fr()
add_all_glc_vol_area_plot(savefig=True)
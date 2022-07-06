#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 19:34:42 2020

@author: keeptg
"""

import os, sys
from shutil import copyfile
import geopandas as gpd
from oggm import cfg, utils, tasks, workflow
from oggm.workflow import execute_entity_task

# from rgitools.funcs import check_geometries
sys.path.append('/home/keeptg/Data/Study_in_Innsbruck/Tienshan/Script')
from Re_GlabTop import prepare_Glabtop_inputs

def run_model(glc_inv_gdf, model_type, working_dir, intersects_dir=None,
              dem_type=None, resetworking_dir=True, use_mp_process=True,
              totle_task=True, **kwargs):

    cfg.initialize()
    if intersects_dir:
        cfg.set_intersects_db(intersects_dir)
    # cfg.PARAMS['border'] = 20
    cfg.PATHS['working_dir'] = utils.mkdir(working_dir, reset=resetworking_dir)
    cfg.PARAMS['use_rgi_area'] = False
    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['dl_verify'] = False
    if not use_mp_process:
        cfg.PARAMS['mp_processes'] = False
    
    gdirs = workflow.init_glacier_directories(glc_inv_gdf)
    if model_type == 'oggm':
        from oggm.tasks import mass_conservation_inversion
        inversion = mass_conservation_inversion
    elif model_type == 'glabtop':
        from Re_GlabTop import base_stress_inversion
        inversion = base_stress_inversion
    else:
        raise ValueError("Wrong model_type: {model_type}")
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
        tasks.prepare_for_inversion,
        prepare_Glabtop_inputs]
    
    task_list2 = [
        inversion,
        tasks.filter_inversion_output,
        tasks.distribute_thickness_per_altitude]
    
    task_list = task_list1 + task_list2 if totle_task else task_list2
    
    execute_entity_task(tasks.define_glacier_region, gdirs, source=dem_type)
    for task in task_list:
        if task == inversion:
            execute_entity_task(task, gdirs, **kwargs)
        else:
            execute_entity_task(task, gdirs)
    
    return gdirs


def main():
    root_dir = '/home/users/lifei'
    data_dir = os.path.join(root_dir, 'Data', 'Tienshan')
    output_dir = '/home/www/lifei/run_output/tienshan_ice_volume'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # gamdam run
    path = os.path.join(data_dir, 'checked_gamdam')
    cdf = gpd.read_file(path)
    work_dir = os.environ["WORKDIR"]
    
    oggm_gam_dir = os.path.join(work_dir, 'oggm_gam')
    oggm_gam_gdirs = run_model(cdf, model_type='oggm', resetworking_dir=True, 
                          working_dir=oggm_gam_dir,
                          intersects_dir=os.path.join(data_dir,
                          'gamdam_intersects'))
    utils.compile_glacier_statistics(oggm_gam_gdirs)
    copyfile(os.path.join(oggm_gam_dir, 'glacier_statistics.csv'),
             os.path.join(output_dir, 'glacier_statistics_oggm_gam.csv'))
     
    glab_gam_dir = os.path.join(work_dir, 'glab_gam')
    glabtop_gam_gdirs = run_model(cdf, model_type='glabtop', resetworking_dir=True,
                          working_dir=glab_gam_dir,
                          intersects_dir=os.path.join(data_dir,
                          'gamdam_intersects'))
    utils.compile_glacier_statistics(glabtop_gam_gdirs)
    copyfile(os.path.join(glab_gam_dir, 'glacier_statistics.csv'),
             os.path.join(output_dir, 'glacier_statistics_glab_gam.csv'))
    
    # rgi run
    cdf = gpd.read_file(os.path.join(data_dir, 'rgi'))
    oggm_rgi_dir = os.path.join(work_dir, 'oggm_rgi')
    oggm_rgi_gdirs = run_model(cdf, model_type='oggm', resetworking_dir=True, 
                          working_dir=oggm_rgi_dir,
                          intersects_dir=os.path.join(data_dir,
                          'gamdam_intersects'))
    utils.compile_glacier_statistics(oggm_rgi_gdirs)
    copyfile(os.path.join(oggm_rgi_dir, 'glacier_statistics.csv'),
             os.path.join(output_dir, 'glacier_statistics_oggm_rgi.csv'))
    
    glab_rgi_dir = os.path.join(work_dir, 'glab_rgi')
    glabtop_rgi_gdirs = run_model(cdf, model_type='glabtop', resetworking_dir=True,
                          working_dir=glab_rgi_dir)
    utils.compile_glacier_statistics(glabtop_rgi_gdirs)
    copyfile(os.path.join(glab_rgi_dir, 'glacier_statistics.csv'),
             os.path.join(output_dir, 'glacier_statistics_glab_rgi.csv'))

if __name__ == '__main__':
    main()


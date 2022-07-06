import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd

import xarray as xr
import salem 

%matplotlib widget
import matplotlib.pyplot as plt

from path_config import *


def get_cmip6_climate():
    if os.path.exists(os.path.join(data_dir, 'statistic_cmip6_climate.pkl')):
        with open(os.path.join(data_dir, 'statistic_cmip6_climate.pkl'), 'rb') as f:
            pr_ssp_cn_list, ts_ssp_cn_list, pr_ssp_fr_list, ts_ssp_fr_list = pickle.load(f)
        return pr_ssp_cn_list, ts_ssp_cn_list, pr_ssp_fr_list, ts_ssp_fr_list
    path = os.path.join(cluster_dir, 'compiling_gcm')
    with open(os.path.join(cluster_dir, 'in_china_list.pkl'), 'rb') as f:
        in_cn = pickle.load(f)
        in_cn = in_cn['ggi_in_China']

    gcms = ['BCC-CSM2-MR', 'CAMS-CSM1-0', 'CESM2', 'CESM2-WACCM',
            'CMCC-CM2-SR5', 'EC-Earth3', 'EC-Earth3-Veg', 'FGOALS-f3-L',
            'GFDL-ESM4', 'INM-CM4-8', 'INM-CM5-0', 'MPI-ESM1-2-HR',
            'MRI-ESM2-0', 'NorESM2-MM', 'TaiESM1']
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

    pr_ssp_cn_list = []
    ts_ssp_cn_list = []
    pr_ssp_fr_list = []
    ts_ssp_fr_list = []
    for ssp in ssps:
        pr_gcm_cn_list = []
        ts_gcm_cn_list = []
        pr_gcm_fr_list = []
        ts_gcm_fr_list = []
        for gcm in gcms:
            fpath = os.path.join(path, gcm+'_'+ssp+'.nc')
            if not os.path.exists(fpath):
                continue
            ds = xr.open_dataset(fpath)
            pr_da = ds.prcp
            ts_da = ds.temp
            prcp_year_cn = pr_da.sel(rgi_id=in_cn)
            temp_year_cn = ts_da.sel(rgi_id=in_cn)
            prcp_year_fr = pr_da.sel(rgi_id=~np.array(in_cn))
            temp_year_fr = ts_da.sel(rgi_id=~np.array(in_cn))
            pr_cn_list = [prcp_year_cn[(prcp_year_cn.hydro_year==year).values, :].sum(dim='time').mean(dim='rgi_id').values.tolist() for year in range(2018, 2100)]
            ts_cn_list = [temp_year_cn[(temp_year_cn.hydro_year==year).values, :].mean(dim='time').mean(dim='rgi_id').values.tolist() for year in range(2018, 2100)]
            pr_fr_list = [prcp_year_fr[(prcp_year_fr.hydro_year==year).values, :].sum(dim='time').mean(dim='rgi_id').values.tolist() for year in range(2018, 2100)]
            ts_fr_list = [temp_year_fr[(temp_year_fr.hydro_year==year).values, :].mean(dim='time').mean(dim='rgi_id').values.tolist() for year in range(2018, 2100)]
            pr_gcm_cn_list.append(pr_cn_list)
            pr_gcm_fr_list.append(pr_fr_list)
            ts_gcm_cn_list.append(ts_cn_list)
            ts_gcm_fr_list.append(ts_fr_list)
            print(f"Finish {ssp} {gcm} !")
        pr_ssp_cn_list.append(pr_gcm_cn_list)
        pr_ssp_fr_list.append(pr_gcm_fr_list)
        ts_ssp_cn_list.append(ts_gcm_cn_list)
        ts_ssp_fr_list.append(ts_gcm_fr_list)

    with open(os.path.join(data_dir, 'statistic_cmip6_climate.pkl'), 'wb') as f:
        pickle.dump([pr_ssp_cn_list, ts_ssp_cn_list, pr_ssp_fr_list, ts_ssp_fr_list], f)
    return pr_ssp_cn_list, ts_ssp_cn_list, pr_ssp_fr_list, ts_ssp_fr_list


def plot(ax, data_list):
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    colors = ['dodgerblue', 'lime', 'orange','darkred']
    for ssp, color, data in zip(ssps, colors, data_list):
        da = np.array(data)
        ax.plot(range(2018, 2100), da.mean(axis=0), lw=1, color=color, label=ssp)


pr_ssp_cn_list, ts_ssp_cn_list, pr_ssp_fr_list, ts_ssp_fr_list = get_cmip6_climate()
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plot(axs[0, 0], pr_ssp_cn_list)
plot(axs[0, 1], pr_ssp_fr_list)
plot(axs[1, 0], ts_ssp_cn_list)
plot(axs[1, 1], ts_ssp_fr_list)
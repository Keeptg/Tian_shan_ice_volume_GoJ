import os
import pickle
import glob
from scipy.ndimage import gaussian_filter1d
import numpy as np
import pandas as pd
import xarray as xr

from path_config import *

import matplotlib.pyplot as plt

def get_gcm_ssp_df(gi_type=None, cmip_path=None, dem_type='SRTM'):

    def _get_gcm_ssp_df(gi_type, cmip_path, dem_type=dem_type):
        path_list = glob.glob(os.path.join(cmip_path, gi_type, dem_type, '*.nc'))
        file_list = [os.path.split(path)[1] for path in path_list]
        gcm_list = list(set([file.split('_', -1)[2] for file in file_list if 'history' not in file]))
        ssp_list = list(set([file.split('_', -1)[3].split('.')[0] 
            for file in file_list if 'history' not in file]))

        list1 = []
        for gcm in gcm_list:
            list2 = []
            for ssp in ssp_list:
                gcm_ssp = gcm + '_' + ssp
                for path in path_list:
                    if gcm_ssp in path:
                        flag2 = 'Y'
                        break
                else:
                    flag2 = ''
                list2.append(flag2)
            list1.append(list2)

        gcm_ssp_df = pd.DataFrame(index=gcm_list, columns=ssp_list, data=list1).sort_index()
        gcm_ssp_df = gcm_ssp_df.reindex(sorted(gcm_ssp_df.columns), axis=1)

        return gcm_ssp_df
    
    if cmip_path is None:
        cmip_path = os.path.join(cluster_dir, 'tienshan_glacier_change', 
                                 'glacier_change_in_21st', 'cmip6')
    if gi_type == None:
        assert np.all(_get_gcm_ssp_df('rgi', cmip_path, dem_type)==
            _get_gcm_ssp_df('ggi', cmip_path, dem_type))
        output = _get_gcm_ssp_df('rgi', cmip_path, dem_type)
        return _get_gcm_ssp_df('rgi', cmip_path, dem_type)
    else:
        return _get_gcm_ssp_df(gi_type, cmip_path, dem_type)


def get_glc_proj_data(ssp, gi_type, var, dem_type, sub_region=None, 
                      fpath=None, fprefix='run_output_'):
    """Compile the projection data as `np.DataFrame`

    Parameters
    ------
    ssp : str:
        the target emmision scenorio
    gi_type : str:
        the target glacier inventory: should be one of ['ggi', 'rgi']
    var : str:
        the target variable: should be one of ['volume', 'area', 'length']
    dem_type : str
        the dem_type used in glacier project simulation
    sub_region : str or None:
        statistics glaciers in subregion or not:
        should be one of [None, 'fr', 'cn'], default None, for all glaciers in Tian Shan
    fpath : str:
        the directory path of projection output `*.nc` file
    fperfix : str:
        the perfix of projection output `*.nc` file

    return
    ------
    output_df : None or py:class: `pd.DataFrame`
    """

    if sub_region:
        with open(os.path.join(cluster_dir, 'in_china_list.pkl'), 'rb') as f:
            in_china = pickle.load(f)
        if gi_type == 'rgi':
            in_rg = np.array(in_china['rgi_in_China'])
        elif gi_type == 'ggi':
            in_rg = np.array(in_china['ggi_in_China'])
        else:
            raise ValueError(f"Unexcepted glacier inventory type: {gi_type}!")
        if sub_region == 'fr':
            in_rg = ~in_rg
        elif sub_region == 'cn':
            pass
        else:
            raise ValueError(f"Unexcepted sub_region: {sub_region}!")

    if fpath is None:
        fpath = os.path.join(cluster_dir, 'tienshan_glacier_change', 
                             'glacier_change_in_21st', 'cmip6')
    gcms = get_gcm_ssp_df(gi_type, fpath, dem_type).index.values
    var_list = []
    valid_gcms = []
    # The end year of cmip6 dataset range from 2099 to 2299, 
    # we here limit the years within [2018, 2099]
    years = range(2018, 2100)
    for gcm in gcms:
        fname = fprefix + gcm + '_' + ssp + '.nc'
        path = os.path.join(fpath, gi_type, dem_type, fname)
        if os.path.exists(path):
            ds = xr.open_dataset(path)
            var_da = ds[var]
            var_da = var_da.sel(time=years)
            if sub_region:
                var_da = var_da.sel(rgi_id=in_rg)
            var_sum = var_da.sum(dim='rgi_id').values
            var_list.append(var_sum)
            valid_gcms.append(gcm)
    if not var_list:
        return
    else:
        var_df = pd.DataFrame(var_list)
        output_df = pd.DataFrame(index=valid_gcms, columns=years, data=var_df.values*1e-9)

        return output_df
    

def plot_cmip(ax, ssps, single_gcm_line='all', single_gcm_line_ssps='all',
              std_shadow_ssps=None, colors=None, in_percent=True,
              delta=False, ls=None, **kwargs):
    """
    """
    if colors is None:
        colors = {'ssp119': 'navy', 'ssp126': 'dodgerblue', 'ssp434': 'g', 'ssp245': 'lime', 
                  'ssp534-over': 'blueviolet', 'ssp460': 'yellow', 'ssp370': 'orange', 'ssp585': 'darkred'}
    for ssp in ssps:
        df_org = get_glc_proj_data(ssp=ssp, **kwargs)
        if delta:
            df_org = df_org - df_org.shift(-1, axis=1)
        if in_percent:
            v0 = df_org.iloc[0, 0]
            df_pct = df_org / v0
            df = df_pct
        else:
            df = df_org
        avg = df.mean()
        std = df.std()
        df.loc['avg'] = avg

        df.loc['ustd'] = avg + std
        df.loc['lstd'] = avg - std
        filter_window = 1
        ax.plot(df.columns, gaussian_filter1d(df.loc['avg'].values, filter_window),
                color=colors[ssp], lw=1.5, zorder=100, ls=ls)
        if single_gcm_line:
            if (ssp in single_gcm_line_ssps) or (single_gcm_line_ssps == 'all'):
                if single_gcm_line == 'all':
                    gcms = df.index.values
                elif type(single_gcm_line) is list:
                    gcms = single_gcm_line
                else: 
                    raise ValueError(f"Unexcepted single_gcm_line type: {single_gcm_line}!")
                for gcm in gcms:
                    ax.plot(df.columns, gaussian_filter1d(df.loc[gcm].values, filter_window),
                            color=colors[ssp], lw=0.2, alpha=.4, zorder=50)
        if std_shadow_ssps:
            if ssp in std_shadow_ssps:
                ax.fill_between(df.columns, gaussian_filter1d(df.loc['lstd'].values, 
                                                              filter_window),
                                gaussian_filter1d(df.loc['ustd'].values, filter_window),
                                color=colors[ssp], lw=0.8, alpha=.2, zorder=70)
   
    return ax


def volume_in_percent_plot():
    gcm_ssp_df = get_gcm_ssp_df()
    fig, axs = plt.subplots(3, 2, figsize=(6, 9))
    axs_flatten = axs.flatten()
    num = 0
    for sub_region in [None, 'cn', 'fr']:
        for gi_type in ['rgi', 'ggi']:
            plot_cmip(axs_flatten[num], ssps=gcm_ssp_df.columns, single_gcm_line='all', 
                      std_shadow_ssps=['ssp126', 'ssp245', 'ssp585'], gi_type=gi_type, var='volume',
                      sub_region=sub_region, dem_type='SRTM')
            num += 1


def area_in_percent_plot():
    gcm_ssp_df = get_gcm_ssp_df()
    fig, axs = plt.subplots(3, 2, figsize=(6, 9))
    axs_flatten = axs.flatten()
    num = 0
    for sub_region in [None, 'cn', 'fr']:
        for gi_type in ['rgi', 'ggi']:
            plot_cmip(axs_flatten[num], ssps=gcm_ssp_df.columns, single_gcm_line='all', 
                      std_shadow_ssps=['ssp126', 'ssp245', 'ssp585'], gi_type=gi_type, var='area',
                      sub_region=sub_region, dem_type='SRTM')
            num += 1


def delta_volume_plot():
    gcm_ssp_df = get_gcm_ssp_df()
    fig, axs = plt.subplots(3, 2, figsize=(6, 9), sharey='row', sharex='col')
    axs_flatten = axs.flatten()
    num = 0
    choosen_ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    for sub_region in [None, 'cn', 'fr']:
        for gi_type in ['rgi', 'ggi']:
            plot_cmip(axs_flatten[num], ssps=choosen_ssps, single_gcm_line=False,
                    std_shadow_ssps=['ssp126', 'ssp585'], gi_type=gi_type, var='volume',
                    sub_region=sub_region, delta=True, in_percent=False, dem_type='SRTM')
            num += 1


def compare_rgi_ggi_projection_plot(compare_gi=False, fig_save=True, plot_real_value=True, var='volume',
                                    share_y_in_subregion=True, adapt_axis_labels=True):
#    compare_gi = True
#    fig_save=False
#    plot_real_value=True
#    share_y_in_subregion=True

    if plot_real_value:
        nrow = 3
    else:
        nrow = 2
    fig, axs = plt.subplots(nrow, 3, figsize=(9, 9), sharex='col')
    #    ssps = gcm_ssp_df.columns
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    if compare_gi:
        num = 0
        row = 0
        for sub_region in [None, 'cn', 'fr']:
            plot_cmip(axs[row, num], ssps=ssps, single_gcm_line=None,
                    std_shadow_ssps=None, gi_type='rgi', var=var,
                    sub_region=sub_region, in_percent=True, ls=(0, (3, 2)), dem_type='SRTM')
            num += 1
        row += 1
        
        if plot_real_value:
            num = 0
            for sub_region in [None, 'cn', 'fr']:
                plot_cmip(axs[row, num], ssps=ssps, single_gcm_line=None,
                        std_shadow_ssps=None, gi_type='rgi', var=var,
                        sub_region=sub_region, in_percent=False, ls=(0, (3, 2)), dem_type='SRTM')
                num += 1
            row += 1

        num = 0
        for sub_region in [None, 'cn', 'fr']:
            plot_cmip(axs[row, num], ssps=ssps, single_gcm_line=None,
                    std_shadow_ssps=None, gi_type='rgi', var=var, ls=(0, (3, 2)),
                    sub_region=sub_region, in_percent=False, delta=True, dem_type='SRTM')
            num += 1

    num = 0
    row = 0
    for sub_region in [None, 'cn', 'fr']:
        plot_cmip(axs[row, num], ssps=ssps, single_gcm_line=None,
                  std_shadow_ssps=['ssp126', 'ssp585'], gi_type='ggi', var=var,
                  sub_region=sub_region, in_percent=True, dem_type='SRTM')
        num += 1
    row += 1

    if plot_real_value:
        num = 0
        for sub_region in [None, 'cn', 'fr']:
            plot_cmip(axs[row, num], ssps=ssps, single_gcm_line=None,
                    std_shadow_ssps=['ssp126', 'ssp585'], gi_type='ggi', var=var,
                    sub_region=sub_region, in_percent=False, dem_type='SRTM')
            num += 1
        row += 1
        
    num = 0
    for sub_region in [None, 'cn', 'fr']:
        plot_cmip(axs[row, num], ssps=ssps, single_gcm_line=None,
                  std_shadow_ssps=['ssp126', 'ssp585'], gi_type='ggi', var=var,
                  sub_region=sub_region, in_percent=False, delta=True, dem_type='SRTM')
        num += 1

    row = 0
    ax0 = axs[row, 0]
    row += 1
    ax0.set_ylabel('Mass (rel. to 2018)')
    x = .95
    y = .98
    diff = .07
    ax0.text(x, y, 'SSP1-2.6', color='dodgerblue', fontsize=10, ha='right',
             va='top', transform=ax0.transAxes, weight='bold')
    y -= diff
    ax0.text(x, y, 'SSP2-4.5', color='lime', fontsize=10, ha='right',
             va='top', transform=ax0.transAxes, weight='bold')
    y -= diff
    ax0.text(x, y, 'SSP3-7.0', color='orange', fontsize=10, ha='right',
             va='top', transform=ax0.transAxes, weight='bold')
    y -= diff
    ax0.text(x, y, 'SSP5-8.5', color='darkred', fontsize=10, ha='right',
             va='top', transform=ax0.transAxes, weight='bold')

    if compare_gi:
        ax0.plot((0, 0), (0, 0), lw=1.5, color='dimgrey', label='GAMDAMv2')
        ax0.plot((0, 0), (0, 0), lw=1.5, ls=(0, (3, 2)), color='dimgrey', label='RGIv6')
        ax0.legend(loc='lower left', fontsize='large', prop=dict(weight='bold'))

    if plot_real_value:
        ax1 = axs[row, 0]
        ax1.set_ylabel('Mass (km$^3$)')
        row += 1
    

    ax1 = axs[row, 0]
    ax1.set_ylabel('Annual mass loss (km$^3$)')
    ax1.set_xlabel('Year')

    [axs[-1, i].set_xlabel('Year') for i in range(0, 3)]
    [ax.tick_params(axis='y', labelrotation=90) for ax in axs.flatten()]
    axs[0, 0].set_yticklabels(axs[0, 0].get_yticks().astype(float))
    axs[0, -1].set_yticklabels(axs[0, -1].get_yticks())
    [axs[0, i].set_title(title, fontsize=10, weight='bold')
        for i, title in zip(range(0, 3), ['Total Tian Shan', 'In China', 'Outside China'])]
    axs[-1, -1].set_yticks([1, 2, 3, 4])
    axs[-1, 0].set_yticks(range(2, 9, 2))
    axs[-1, 0].set_yticklabels(range(1, 10, 2))
    [ax.set_ylim(.1, 1.) for ax in axs[0, :]]
    [ax.set_yticks([.2, .4, .6, .8, 1.]) for ax in axs[0, :]]
    [ax.set_yticklabels([.2, .4, .6, .8, 1.]) for ax in axs[0, :]]
    if plot_real_value:
        axs[1, 1].set_yticks([100, 200, 300])
        axs[1, 2].set_yticks([100, 200, 300])

    label_list = [chr(asc) for asc in range(65, 65+len(axs.flatten())+1)]
    for j, (i, ax) in enumerate(zip(label_list, axs.flatten())):
        ax.text(.99, .01, i, transform=ax.transAxes, weight='bold',
                ha='right', va='bottom', fontsize=10)
        ax.set_xlim(2018, 2098)
        if j < 3:
            ax.set_ylim(0, 1)

    [ax.set_xticks([2030, 2060, 2090]) for ax in axs[2, :]]
    plt.tight_layout()
    plt.subplots_adjust(hspace=.03)
    if share_y_in_subregion:
        [ax.tick_params(axis='y', right=True, left=False, labelright=False, labelleft=False) for ax in axs[:, 1]]
        [ax.tick_params(axis='y', right=True, left=False, labelright=True, labelleft=False) for ax in axs[:, 2]]
        [ax.yaxis.set_label_position('right') for ax in axs[:, 2]]
        if plot_real_value:
            [ax.set_ylim(25, 325) for ax in axs[1, 1:3]]
            [ax.set_yticklabels([100, 200, 300]) for ax in axs[1, 1:3]]
        [ax.set_ylim(.2, 5) for ax in axs[-1, 1:3]]
        [ax.set_yticks(range(1, 6, 1)) for ax in axs[-1, 1:3]]
        [ax.set_yticklabels(range(1, 6, 1)) for ax in axs[-1, 1:3]]
        [ax.set_ylabel(ylabel)
            for ylabel, ax in zip(['Mass (rel. 2018)', 'Mass (km$^3$)', 'Annual mass loss (km$^3$)'], 
                                  axs[:, 2])]
        for ax in axs[:, 2]:
            opos = ax.get_position()
            mpos = [opos.x0-.032, opos.y0, opos.width, opos.height]
            ax.set_position(mpos)


    if fig_save:
        fig.savefig(os.path.join(data_dir, 'figure', 'compare_rgi_ggi_glc_change.pdf'), 
                    dpi=600, bbox_inches='tight')

    return fig, axs


def compare_none_specific_regional_plot(ax=None, save_fig=False, legend_fontsize=12):
    if ax is None:
        save_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.set_ylim(0, 1)
        ax.set_ylabel('Mass (rel. 2018)')
        ax.set_xlabel('Year')
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    plot_cmip(ax, ssps=ssps, single_gcm_line=False, single_gcm_line_ssps=False,
              dem_type='SRTM_none_geodetic_calib', gi_type='rgi', var='volume', ls=(0, (6, 3)))
    plot_cmip(ax, ssps=ssps, single_gcm_line=False, single_gcm_line_ssps=False,
              dem_type='SRTM_specific_geodetic_calib', gi_type='rgi', var='volume', ls=(0, (2, 2)))
    plot_cmip(ax, ssps=ssps, single_gcm_line=False, single_gcm_line_ssps=False,
              dem_type='SRTM', gi_type='rgi', var='volume')
    x = .95
    y = .98
    diff = .05
    ax.text(x, y, 'SSP1-2.6', color='dodgerblue', fontsize=legend_fontsize, ha='right', va='top', transform=ax.transAxes)
    y -= diff
    ax.text(x, y, 'SSP2-4.5', color='lime', fontsize=legend_fontsize, ha='right', va='top', transform=ax.transAxes)
    y -= diff
    ax.text(x, y, 'SSP3-7.0', color='orange', fontsize=legend_fontsize, ha='right', va='top', transform=ax.transAxes)
    y -= diff
    ax.text(x, y, 'SSP5-8.5', color='darkred', fontsize=legend_fontsize, ha='right', va='top', transform=ax.transAxes)
    ax.plot((0, 0), (0, 0), color='dimgrey', lw=1.5, label='regional')
    ax.plot((0, 0), (0, 0), color='dimgrey', lw=1.5, ls=(0, (2, 2)), label='specific')
    ax.plot((0, 0), (0, 0), color='dimgrey', lw=1.5, ls=(0, (6, 3)), label='none')
    ax.legend(loc='lower left', fontsize=legend_fontsize)
    ax.set_xlim(2018, 2099)
    plt.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(data_dir, 'figure', 'compare_non_specific_regional_glc_change.pdf'), dpi=300,
                    bbox_inches='tight')


def compare_glen_a_plot(ax=None, save_fig=False, legend_fontsize=12, weight=None):
    if ax is None:
        save_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.set_ylim(0, 1)
        ax.set_ylabel('Mass (rel. 2018)')
        ax.set_xlabel('Year')
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    plot_cmip(ax, ssps=ssps, single_gcm_line=False, single_gcm_line_ssps=False,
              dem_type='SRTM_(1.93)', gi_type='ggi', var='volume', ls=(0, (6, 3)))
    plot_cmip(ax, ssps=ssps, single_gcm_line=False, single_gcm_line_ssps=False,
              dem_type='SRTM_(6.18)', gi_type='ggi', var='volume', ls=(0, (2, 2)))
    plot_cmip(ax, ssps=ssps, single_gcm_line=False, single_gcm_line_ssps=False,
              dem_type='SRTM', gi_type='ggi', var='volume')
    x = .95
    y = .98
    diff = .05
    ax.text(x, y, 'SSP1-2.6', color='dodgerblue', fontsize=legend_fontsize,
            ha='right', va='top', transform=ax.transAxes, weight=weight)
    y -= diff
    ax.text(x, y, 'SSP2-4.5', color='lime', fontsize=legend_fontsize,
            ha='right', va='top', transform=ax.transAxes, weight=weight)
    y -= diff
    ax.text(x, y, 'SSP3-7.0', color='orange', fontsize=legend_fontsize,
            ha='right', va='top', transform=ax.transAxes, weight=weight)
    y -= diff
    ax.text(x, y, 'SSP5-8.5', color='darkred', fontsize=legend_fontsize,
            ha='right', va='top', transform=ax.transAxes, weight=weight)
    ax.plot((0, 0), (0, 0), color='dimgrey', lw=1.5, 
            label=r'opt ($5.30 \times A_{def}$)')
    ax.plot((0, 0), (0, 0), color='dimgrey', lw=1.5, ls=(0, (2, 2)), 
            label=r'max ($6.18 \times A_{def}$)')
    ax.plot((0, 0), (0, 0), color='dimgrey', lw=1.5, ls=(0, (6, 3)),
            label=r'min ($1.93 \times A_{def}$)')
    ax.legend(loc='lower left', fontsize=legend_fontsize)
    ax.set_xlim(2018, 2099)
    plt.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(data_dir, 'figure', 'compare_non_specific_regional_glc_change.pdf'), 
                    dpi=300, bbox_inches='tight')
 

def compare_gi_dem_glc_vol_diff():
    glc_vol_path = os.path.join(cluster_dir, 'tienshan_ice_volume', 'Tienshan_glc_volume_uncertainty')
    glab_ggi_srtm = pd.read_csv(os.path.join(glc_vol_path, 'glab_ggi_SRTM_opt(0.65).csv'))
    glab_ggi_copdem = pd.read_csv(os.path.join(glc_vol_path, 'glab_ggi_COPDEM_opt(0.65).csv'))
    glab_rgi_copdem = pd.read_csv(os.path.join(glc_vol_path, 'glab_rgi_COPDEM_opt(0.65).csv'))
    glab_rgi_srtm = pd.read_csv(os.path.join(glc_vol_path, 'glab_rgi_SRTM_opt(0.65).csv'))
    oggm_ggi_srtm = pd.read_csv(os.path.join(glc_vol_path, 'oggm_ggi_SRTM_opt(5.30).csv'))
    oggm_rgi_srtm = pd.read_csv(os.path.join(glc_vol_path, 'oggm_rgi_SRTM_opt(5.30).csv'))
    oggm_ggi_copdem = pd.read_csv(os.path.join(glc_vol_path, 'oggm_ggi_COPDEM_opt(5.30).csv'))
    oggm_rgi_copdem = pd.read_csv(os.path.join(glc_vol_path, 'oggm_rgi_COPDEM_opt(5.30).csv'))

    glab_ggi_rel_diff = (glab_ggi_srtm.inv_volume_km3 - glab_ggi_copdem.inv_volume_km3)/glab_ggi_srtm.inv_volume_km3
    glab_rgi_rel_diff = (glab_rgi_srtm.inv_volume_km3 - glab_rgi_copdem.inv_volume_km3)/glab_rgi_srtm.inv_volume_km3
    oggm_ggi_rel_diff = (oggm_ggi_srtm.inv_volume_km3 - oggm_ggi_copdem.inv_volume_km3)/oggm_ggi_srtm.inv_volume_km3
    oggm_rgi_rel_diff = (oggm_rgi_srtm.inv_volume_km3 - oggm_rgi_copdem.inv_volume_km3)/oggm_rgi_srtm.inv_volume_km3
    

    with open(os.path.join(cluster_dir, 'in_china_list.pkl'), 'rb') as f:
        in_china = pickle.load(f)
    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    axs[0, 0].hist(glab_ggi_rel_diff, bins=200)
    axs[0, 0].set_xlim(-2, 2)
    axs[0, 1].hist(glab_rgi_rel_diff, bins=200)
    axs[0, 1].set_xlim(-2, 2)
    axs[1, 0].hist(oggm_ggi_rel_diff, bins=200)
    axs[1, 0].set_xlim(-2, 2)
    axs[1, 1].hist(oggm_rgi_rel_diff, bins=200)
    axs[1, 1].set_xlim(-2, 2)


    fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    axs[0, 0].scatter(glab_ggi_srtm.inv_volume_km3, glab_ggi_copdem.inv_volume_km3,
                      c=in_china['ggi_in_China'])
    axs[0, 1].scatter(glab_rgi_srtm.inv_volume_km3, glab_rgi_copdem.inv_volume_km3,
                      c=in_china['rgi_in_China'])
    axs[1, 0].scatter(oggm_ggi_srtm.inv_volume_km3, oggm_ggi_copdem.inv_volume_km3, 
                      c=in_china['ggi_in_China'])
    axs[1, 1].scatter(oggm_rgi_srtm.inv_volume_km3, oggm_rgi_copdem.inv_volume_km3,
                      c=in_china['rgi_in_China'])
    for ax in axs.flatten():
        ax.plot((0, 90), (0, 90), ls=(0, (2, 2)), lw=.5, color='black')


def compare_dem_plot(ax=None, legend_fontsize=12, save_fig=False, weight=None):
    if ax is None:
        save_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.set_ylim(0, 1)
        ax.set_ylabel('Mass (rel. 2018)')
        ax.set_xlabel('Year')
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    plot_cmip(ax, ssps=ssps, single_gcm_line=False, single_gcm_line_ssps=False,
              dem_type='COPDEM', gi_type='ggi', var='volume', ls=(0, (6, 3)))
    plot_cmip(ax, ssps=ssps, single_gcm_line=False, single_gcm_line_ssps=False,
              dem_type='SRTM', gi_type='ggi', var='volume')
    x = .95
    y = .98
    diff = .05
    ax.text(x, y, 'SSP1-2.6', color='dodgerblue', fontsize=legend_fontsize,
            ha='right', va='top', transform=ax.transAxes, weight=weight)
    y -= diff
    ax.text(x, y, 'SSP2-4.5', color='lime', fontsize=legend_fontsize,
            ha='right', va='top', transform=ax.transAxes, weight=weight)
    y -= diff
    ax.text(x, y, 'SSP3-7.0', color='orange', fontsize=legend_fontsize,
            ha='right', va='top', transform=ax.transAxes, weight=weight)
    y -= diff
    ax.text(x, y, 'SSP5-8.5', color='darkred', fontsize=legend_fontsize,
            ha='right', va='top', transform=ax.transAxes, weight=weight)
    ax.plot((0, 0), (0, 0), color='dimgrey', lw=1.5, 
            label=r'SRTM')
    ax.plot((0, 0), (0, 0), color='dimgrey', lw=1.5, ls=(0, (6, 3)),
            label=r'COPDEM')
    ax.legend(loc='lower left', fontsize=legend_fontsize)
    ax.set_xlim(2018, 2099)
    plt.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(data_dir, 'figure', 'compare_dem_plot.pdf'), 
                    dpi=300, bbox_inches='tight')


def compare_gi_plot(ax=None, legend_fontsize=12, save_fig=False, weight=None):
    if ax is None:
        save_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.set_ylim(0, 1)
        ax.set_ylabel('Mass (rel. 2018)')
        ax.set_xlabel('Year')
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    plot_cmip(ax, ssps=ssps, single_gcm_line=False, single_gcm_line_ssps=False,
              dem_type='SRTM', gi_type='rgi', var='volume', ls=(0, (6, 3)))
    plot_cmip(ax, ssps=ssps, single_gcm_line=False, single_gcm_line_ssps=False,
              dem_type='SRTM', gi_type='ggi', var='volume')
    x = .95
    y = .98
    diff = .05
    ax.text(x, y, 'SSP1-2.6', color='dodgerblue', fontsize=legend_fontsize,
            ha='right', va='top', transform=ax.transAxes, weight=weight)
    y -= diff
    ax.text(x, y, 'SSP2-4.5', color='lime', fontsize=legend_fontsize,
            ha='right', va='top', transform=ax.transAxes, weight=weight)
    y -= diff
    ax.text(x, y, 'SSP3-7.0', color='orange', fontsize=legend_fontsize,
            ha='right', va='top', transform=ax.transAxes, weight=weight)
    y -= diff
    ax.text(x, y, 'SSP5-8.5', color='darkred', fontsize=legend_fontsize,
            ha='right', va='top', transform=ax.transAxes, weight=weight)
    ax.plot((0, 0), (0, 0), color='dimgrey', lw=1.5, 
            label=r'GAMDAMv2')
    ax.plot((0, 0), (0, 0), color='dimgrey', lw=1.5, ls=(0, (6, 3)),
            label=r'RGIv6')
    ax.legend(loc='lower left', fontsize=legend_fontsize)
    ax.set_xlim(2018, 2099)
    plt.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(data_dir, 'figure', 'compare_dem_plot.pdf'), 
                    dpi=300, bbox_inches='tight')


def plot_sensitive(legend_fontsize=10, save_fig=False, weight=None):
#    fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.5, 3.5), sharey=True)
#    compare_none_specific_regional_plot(ax=ax0, legend_fontsize=legend_fontsize)
    compare_dem_plot(ax=ax0, legend_fontsize=legend_fontsize, weight=weight)
#    compare_gi_plot(ax=ax1, legend_fontsize=legend_fontsize, weight=weight)
    compare_glen_a_plot(ax=ax1, legend_fontsize=legend_fontsize, weight=weight)
    ax0.set_ylim(0, 1)
    ax0.set_ylabel('Mass (rel. to 2018)')
    for ax in [ax0, ax1]:
        ax.set_xlabel('Year')
    plt.tight_layout()
    ax0.text(.99, .01, 'A', ha='right', va='bottom', transform=ax0.transAxes, weight='bold')
    ax1.text(.99, .01, 'B', ha='right', va='bottom', transform=ax1.transAxes, weight='bold')
    plt.tight_layout()
    if save_fig:
        fig.savefig(os.path.join(data_dir, 'figure', 'glc_chg_sensitive.pdf'),
                    dpi=300, bbox_inches='tight')
    
    
def output_numbers():
    def print_number(ssp, sub_region, year, gi_type='ggi', var='volume',
                     dem_type='SRTM', normalized=True):
        df = get_glc_proj_data(ssp=ssp, gi_type=gi_type, var=var,
                               dem_type=dem_type, sub_region=sub_region)
        avg = df.mean()
        std = df.std()
        df.loc['mean'] = avg
        df.loc['std'] = std
        if normalized:
            df = df / df.iloc[0, 0]
        print("{:.0%} +/- {:.0%}".format(df[year]['mean'], df[year]['std']))
    
    for ssp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
        print_number(ssp, None, 2050)
    for ssp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
        print_number(ssp, None, 2099)
    for ssp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
        print_number(ssp, 'cn', 2050)
    for ssp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
        print_number(ssp, 'cn', 2099)
    for ssp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
        print_number(ssp, 'fr', 2050)
    for ssp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
        print_number(ssp, 'fr', 2099)


def statistics_mass_diff_rgi_ggi():

    def calculate_diff_with_uncertainty(df1, df2):
        diff = (df1.mean().iloc[0] - df1.mean().iloc[-1] - (df2.mean().iloc[0] - df2.mean().iloc[1]))
        diff = (df1[2018] - df1[2099]) - (df2[2018] - df2[2099])
        mean = diff.mean()
        std = diff.std()
        print("{:.2f}+-{:.2f}".format(mean, std))
        
    cn_ggi_26 = get_glc_proj_data('ssp126', 'ggi', 'volume', 'SRTM', sub_region='cn')
    cn_rgi_26 = get_glc_proj_data('ssp126', 'rgi', 'volume', 'SRTM', sub_region='cn')
    cn_ggi_45 = get_glc_proj_data('ssp245', 'ggi', 'volume', 'SRTM', sub_region='cn')
    cn_rgi_45 = get_glc_proj_data('ssp245', 'rgi', 'volume', 'SRTM', sub_region='cn')
    cn_ggi_70 = get_glc_proj_data('ssp370', 'ggi', 'volume', 'SRTM', sub_region='cn')
    cn_rgi_70 = get_glc_proj_data('ssp370', 'rgi', 'volume', 'SRTM', sub_region='cn')
    cn_ggi_85 = get_glc_proj_data('ssp585', 'ggi', 'volume', 'SRTM', sub_region='cn')
    cn_rgi_85 = get_glc_proj_data('ssp585', 'rgi', 'volume', 'SRTM', sub_region='cn')

    fr_ggi_26 = get_glc_proj_data('ssp126', 'ggi', 'volume', 'SRTM', sub_region='fr')
    fr_rgi_26 = get_glc_proj_data('ssp126', 'rgi', 'volume', 'SRTM', sub_region='fr')
    fr_ggi_45 = get_glc_proj_data('ssp245', 'ggi', 'volume', 'SRTM', sub_region='fr')
    fr_rgi_45 = get_glc_proj_data('ssp245', 'rgi', 'volume', 'SRTM', sub_region='fr')
    fr_ggi_70 = get_glc_proj_data('ssp370', 'ggi', 'volume', 'SRTM', sub_region='fr')
    fr_rgi_70 = get_glc_proj_data('ssp370', 'rgi', 'volume', 'SRTM', sub_region='fr')
    fr_ggi_85 = get_glc_proj_data('ssp585', 'ggi', 'volume', 'SRTM', sub_region='fr')
    fr_rgi_85 = get_glc_proj_data('ssp585', 'rgi', 'volume', 'SRTM', sub_region='fr')

    for df1, df2 in zip([cn_ggi_26, cn_ggi_45, cn_ggi_70, cn_ggi_85],
                        [cn_rgi_26, cn_rgi_45, cn_rgi_70, cn_rgi_85]):
        calculate_diff_with_uncertainty(df1, df2)
    
    
    for df1, df2 in zip([fr_ggi_26, fr_ggi_45, fr_ggi_70, fr_ggi_85],
                        [fr_rgi_26, fr_rgi_45, fr_rgi_70, fr_rgi_85]):
        calculate_diff_with_uncertainty(df1, df2)

    cn_ggi_26_diff = cn_ggi_26.mean().iloc[0] - cn_ggi_26.mean().iloc[-1]-(cn_rgi_26.mean().iloc[0] - cn_rgi_26.mean().iloc[-1])
    cn_ggi_45_diff = cn_ggi_45.mean().iloc[0] - cn_ggi_45.mean().iloc[-1]-(cn_rgi_45.mean().iloc[0] - cn_rgi_45.mean().iloc[-1])
    cn_ggi_70_diff = cn_ggi_70.mean().iloc[0] - cn_ggi_70.mean().iloc[-1]-(cn_rgi_70.mean().iloc[0] - cn_rgi_70.mean().iloc[-1])
    cn_ggi_85_diff = cn_ggi_85.mean().iloc[0] - cn_ggi_85.mean().iloc[-1]-(cn_rgi_85.mean().iloc[0] - cn_rgi_85.mean().iloc[-1])
    fr_ggi_26_diff = fr_ggi_26.mean().iloc[0] - fr_ggi_26.mean().iloc[-1]-(fr_rgi_26.mean().iloc[0] - fr_rgi_26.mean().iloc[-1])
    fr_ggi_45_diff = fr_ggi_45.mean().iloc[0] - fr_ggi_45.mean().iloc[-1]-(fr_rgi_45.mean().iloc[0] - fr_rgi_45.mean().iloc[-1])
    fr_ggi_70_diff = fr_ggi_70.mean().iloc[0] - fr_ggi_70.mean().iloc[-1]-(fr_rgi_70.mean().iloc[0] - fr_rgi_70.mean().iloc[-1])
    fr_ggi_85_diff = fr_ggi_85.mean().iloc[0] - fr_ggi_85.mean().iloc[-1]-(fr_rgi_85.mean().iloc[0] - fr_rgi_85.mean().iloc[-1])

    cn_ggi_26_45_diff_mean = cn_ggi_26.mean().iloc[0] - cn_ggi_26.mean().iloc[-1] -\
        (cn_ggi_45.mean().iloc[0] - cn_ggi_45.mean().iloc[-1])
    cn_ggi_45_70_diff_mean = cn_ggi_45.mean().iloc[0] - cn_ggi_45.mean().iloc[-1] -\
        (cn_ggi_70.mean().iloc[0] - cn_ggi_70.mean().iloc[-1])
    cn_ggi_70_85_diff_mean = cn_ggi_70.mean().iloc[0] - cn_ggi_70.mean().iloc[-1] -\
        (cn_ggi_85.mean().iloc[0] - cn_ggi_85.mean().iloc[-1])

    fr_ggi_26_45_diff_mean = fr_ggi_26.mean().iloc[0] - fr_ggi_26.mean().iloc[-1] -\
        (fr_ggi_45.mean().iloc[0] - fr_ggi_45.mean().iloc[-1])
    fr_ggi_45_70_diff_mean = fr_ggi_45.mean().iloc[0] - fr_ggi_45.mean().iloc[-1] -\
        (fr_ggi_70.mean().iloc[0] - fr_ggi_70.mean().iloc[-1])
    fr_ggi_70_85_diff_mean = fr_ggi_70.mean().iloc[0] - fr_ggi_70.mean().iloc[-1] -\
        (fr_ggi_85.mean().iloc[0] - fr_ggi_85.mean().iloc[-1])

    cn_ggi_26_45_diff = cn_ggi_26.iloc[:, 0] - cn_ggi_26.iloc[:, -1] -\
        (cn_ggi_45.iloc[:, 0] - cn_ggi_45.iloc[:, -1])
    cn_ggi_45_70_diff = cn_ggi_45.iloc[:, 0] - cn_ggi_45.iloc[:, -1] -\
        (cn_ggi_70.iloc[:, 0] - cn_ggi_70.iloc[:, -1])
    cn_ggi_70_85_diff = cn_ggi_70.iloc[:, 0] - cn_ggi_70.iloc[:, -1] -\
        (cn_ggi_85.iloc[:, 0] - cn_ggi_85.iloc[:, -1])

    fr_ggi_26_45_diff = fr_ggi_26.iloc[:, 0] - fr_ggi_26.iloc[:, -1] -\
        (fr_ggi_45.iloc[:, 0] - fr_ggi_45.iloc[:, -1])
    fr_ggi_45_70_diff = fr_ggi_45.iloc[:, 0] - fr_ggi_45.iloc[:, -1] -\
        (fr_ggi_70.iloc[:, 0] - fr_ggi_70.iloc[:, -1])
    fr_ggi_70_85_diff = fr_ggi_70.iloc[:, 0] - fr_ggi_70.iloc[:, -1] -\
        (fr_ggi_85.iloc[:, 0] - fr_ggi_85.iloc[:, -1])

    fr_ggi_26_45_diff.mean()
    fr_ggi_26_45_diff.std()
    fr_ggi_45_70_diff.mean()
    fr_ggi_45_70_diff.std()
    fr_ggi_70_85_diff.mean()
    fr_ggi_70_85_diff.std()

    cn_ggi_26_45_diff.mean()
    cn_ggi_26_45_diff.std()
    cn_ggi_45_70_diff.mean()
    cn_ggi_45_70_diff.std()
    cn_ggi_70_85_diff.mean()
    cn_ggi_70_85_diff.std()
    
def get_residual():
    fpath = os.path.join(cluster_dir, 'tienshan_glacier_change', 'glacier_change_in_21st')
    residual_ggi = pd.read_csv(os.path.join(fpath, 'cmip6', 'ggi', 'SRTM', 'glacier_statistics.csv'))
    residual_rgi = pd.read_csv(os.path.join(fpath, 'cmip6', 'rgi', 'SRTM', 'glacier_statistics.csv'))
    residual_ggi.dropna(axis=0, inplace=True, subset=['mb_bias'])
    residual_rgi.dropna(axis=0, inplace=True, subset=['mb_bias'])
    ggi_residual = np.average(residual_ggi.mb_bias.values, weights=residual_ggi.rgi_area_km2.values)
    rgi_residual = np.average(residual_rgi.mb_bias.values, weights=residual_rgi.rgi_area_km2.values)
#cmip_path = os.path.join(cluster_dir, 'tienshan_glacier_change', 
#                         'glacier_change_in_21st', 'cmip6')
fig, axs = compare_rgi_ggi_projection_plot(fig_save=False, compare_gi=True)
fig, axs = compare_rgi_ggi_projection_plot(fig_save=False, compare_gi=True, var='area')
#plot_sensitive(save_fig=True)
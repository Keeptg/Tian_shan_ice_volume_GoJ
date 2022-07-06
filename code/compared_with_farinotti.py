#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:29:50 2020

@author: keeptg
"""

import os, sys, glob, pickle
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import geopandas as gpd
import salem
from Re_GlabTop import _extract_farinotti_thick
from Re_GlabTop import get_farinotti_file_path
from path_config import root_dir, data_dir, out_dir, cluster_dir

%matplotlib widget
import matplotlib.pyplot as plt
import matplotlib.legend as mlegend
import seaborn as sns


#root_dir = '/home/keeptg/Data/Study_in_Innsbruck'
#data_dir = os.path.join(root_dir, 'Data', 'Shpfile', 'Tienshan_data')
dir_ = os.path.join(root_dir, 'Data', 'cluster_output')

def plot_compare_boxplot(axs=None, savefig=False, meas_point='tester',
                         plot_glc='specific', err_type='normalized'):
#    Test
#    plot_glc='specific'
#    err_type = 'normalized'
#    axs=None
#    savefig=False
#    meas_point='tester'
        
    rgi_gtd_df = gpd.read_file(os.path.join(data_dir, 'rgi_glathida'))
    if meas_point == 'tester':
        gtd_path = os.path.join(root_dir, 'Data', 'model_output', 'ref_glc_validate_tienshan', 'testers.shp')
    elif meas_point == 'trainer':
        gtd_path = os.path.join(root_dir, 'Data', 'model_output', 'ref_glc_validate_tienshan', 'trainers.shp')
    elif meas_point == 'all':
        gtd_path = os.path.join(data_dir, 'glathida2008_2014')
    else:
        raise ValueError("No measure point type named as {}!".format(meas_point))
    gtd_df = gpd.read_file(os.path.join(gtd_path))

    fr_model_types = ['0', '1', '2', '3', '4']

    fr_paths = [[get_farinotti_file_path(rgiid, model_type) for 
                 model_type in fr_model_types] for rgiid in rgi_gtd_df.RGIId]

    glc_name = rgi_gtd_df.name_1.values
    thick_list = [[_extract_farinotti_thick(gtd_df, path, thick_col='thickness')
                   for path in paths] for paths in fr_paths]

    gtd_fr_thick_dict = dict(zip(glc_name, thick_list))
    calib_data = pd.read_csv(os.path.join(data_dir, 'calibrate_model.csv'))
    p_opt_data = pd.read_csv(os.path.join(data_dir, 'ref_glc_p_opt_mea_cal_thi.csv'))
    p_opt_data['glc_name'] = p_opt_data.glc_name.str.upper()
    p_opt_data['model_type'] = p_opt_data.model_type + '_opt'
    calib_data.drop('Unnamed: 0', axis=1, inplace=True)
    p_opt_data.drop('Unnamed: 0', axis=1, inplace=True)
    _dict = {}
    for glc in list(set(p_opt_data.glc_name)):
        _df = p_opt_data[p_opt_data.glc_name==glc]
        _dict[glc] = _df.mea_thick.mean()

    mean_obs_list = []
    for glc_name in p_opt_data.glc_name.values:
        mean_obs_list.append(_dict[glc_name])
    p_opt_data['mean_obs_thick'] = mean_obs_list
        

    full_name = list(set(calib_data.glc_name))
    name_dic = dict(zip(['Haxilegeng52', 'Heigou8', 'Sarytor', 'Tuyuksu', 'Sigonghe4',
                         'Urumuqi1E', 'Qingbingtan72', 'Urumuqi1W'],
                        ['HXLG', 'HG', 'ST', 'TYK', 'SGH', 'UME', 'QBT', 'UMW']))

    calib_name = [name_dic[fname] for fname in calib_data.glc_name.values]
    calib_data['glc_name'] = calib_name

    mdlist, cdlist, mnlist, gnlist, mob_list = [], [], [], [], []
    for gn, tdata in list(gtd_fr_thick_dict.items()):
        for mn, dt in zip(fr_model_types, tdata):
            gnlist += np.full(len(dt[0]), gn).tolist()
            mdlist += dt[0].tolist()
            cdlist += dt[1].tolist()
            mnlist += np.full(len(dt[0]), mn).tolist()
            mob_list += np.full(len(dt[0]), np.mean(dt[0])).tolist()

    temdf = pd.DataFrame({'mea_thick': mdlist, 'cal_thick': cdlist, 'model_type': mnlist,
                          'glc_name': gnlist, 'mean_obs_thick': mob_list})
    temdf = temdf[temdf.cal_thick != 0]
    temdf['thick_diff'] = temdf.cal_thick - temdf.mea_thick
#    thick_df = pd.concat([calib_data, p_opt_data, temdf], sort=False)
    thick_df = pd.concat([p_opt_data, temdf], sort=False)
    thick_df.to_csv(os.path.join(data_dir, 'calib_oggm_glab_farinotti_thick.csv'))
#   thick_df = thick_df[thick_df.model_type != '0']
    thick_df['thick_diff_abs'] = thick_df.thick_diff.abs()
    thick_df['thick_diff_relative'] = thick_df.thick_diff.values / thick_df.mea_thick.values
    thick_df['thick_diff_normalized'] = thick_df.thick_diff.values / thick_df.mean_obs_thick.values
    flierprops = dict(marker='x', markersize=2, alpha=1, color='grey')
    boxcolor = ['skyblue', 'lightcoral', 'navajowhite', 'gray', 'silver', 'lightgrey', 'whitesmoke']
    hue_order = ['oggm_opt', 'glab_opt', '0', '1', '2', '3', '4']
    order = ['ST', 'TYK', 'HG', 'HXLG', 'SGH', 'UMW', 'UME', 'QBT']
    if err_type == 'relative':
        err = 'thick_diff_relative'
    elif err_type == 'normalized':
        err = 'thick_diff_normalized'
    else:
        err = 'thick_diff'

    if plot_glc == 'specific':
        fig, axs = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
        gtd2_glc = ['ST', 'SGH', 'UMW', 'UME']
        gtd3_glc = ['TYK', 'HG', 'HXLG', 'QBT']

    #    order = ['oggm_opt', 'glab_opt', 'oggm', 'glabtop1', '1', '2', '3', '4']
        sns.boxplot(y=err, hue='model_type', data=thick_df, x='glc_name', ax=axs[0], palette=boxcolor,
                    flierprops=flierprops, hue_order=hue_order, order=gtd2_glc)
        sns.boxplot(y=err, hue='model_type', data=thick_df, x='glc_name', ax=axs[1], palette=boxcolor,
                    flierprops=flierprops, hue_order=hue_order, order=gtd3_glc)
        for ax in axs:
            ax.get_legend().remove()
        ylim1 = ax.get_ylim()
        xlim1 = axs[0].get_xlim()
        axs[0].plot(xlim1, (0, 0), lw=.8, ls=(0, (3, 3)), color='black', zorder=100)
        axs[1].plot(xlim1, (0, 0), lw=.8, ls=(0, (3, 3)), color='black', zorder=100)
        axs[0].set_xlim(xlim1)
        axs[1].set_xlim(xlim1)
        if err_type == 'abs':
            axs[0].set_ylim(-200, 200)
            axs[0].set_yticks(range(-200, 201, 100))
        else:
            axs[0].set_ylim(-5, 5)
            axs[0].set_yticks(range(-5, 6, 2))
            axs[0].set_yticklabels(['-500', '-300', '-100', '100', '300', '500'], rotation=90, va='center')
        axs[0].set_ylabel('Simulation bias (%)')
        axs[0].set_xlabel('')
        axs[0].tick_params(axis='both', which='major', labelsize=9)
        axs[0].text(1, -.12, 'Glacier', fontsize=10, ha='center', va='top', transform=axs[0].transAxes)
        axs[1].set_xlabel('')
        axs[1].spines['left'].set_linewidth(0)
        axs[1].tick_params(axis='y', which='both', left=False)
        axs[1].set_ylabel('')
        axs[1].set_xticklabels(gtd3_glc, weight='bold')
        plt.tight_layout()
        plt.subplots_adjust(wspace=0)
        ax1 = axs[0]
        l_mc = ax1.bar(0, 0, color='skyblue', label='MC', ec='#505050', lw=1.5)
        l_bs = ax1.bar(0, 0, color='lightcoral', label='BS', ec='#505050', lw=1.5)
        l_m0, = ax1.bar(0, 0, color='navajowhite', label='FC ', ec='#505050', lw=1.5)
        l_m1, = ax1.bar(0, 0, color='gray', label='F1', ec='#505050', lw=1.5)
        l_m2, = ax1.bar(0, 0, color='silver', label='F2', ec='#505050', lw=1.5)
        l_m3, = ax1.bar(0, 0, color='lightgrey', label='F3', ec='#505050', lw=1.5)
        l_m4, = ax1.bar(0, 0, color='whitesmoke', label='F4', ec='#505050', lw=1.5)
        leg1 = ax1.legend(handles=[l_mc, l_bs], labels=['MC', 'BS'], ncol=2, columnspacing=1, loc='lower left')
        leg2 = mlegend.Legend(ax1, [l_m0, l_m1, l_m2, l_m3, l_m4],
                            labels=['FC ', 'F1', 'F2', 'F3', 'F4'], columnspacing=1, ncol=5)
        leg1._legend_box._children.append(leg2._legend_box._children[1])
        leg1._legend_box.align='left'
        if savefig:
            plt.savefig(os.path.join(data_dir, 'figure', 'mc_bs_opt_fr_abs_err.pdf'), 
                        dpi=300, bbox_inches='tight')

    if plot_glc == 'entity':
        fig, ax2 = plt.subplots()
        palette = sns.color_palette(boxcolor)
        sns.boxplot(y='thick_diff', x='model_type', data=thick_df, ax=ax2, palette=palette, flierprops=flierprops)
        ax2.set_ylim(-200, 200)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Error (m)')
        ax2.set_xticklabels(['BS_opt', 'MC_opt', 'FC', 'F1', 'F2', 'F3', 'F4'])
        ax2.tick_params(axis='both', which='major', labelsize=9)
    #    ax1.text(0.01, .97, 'A', ha='left', va='top', fontweight='bold', transform=ax1.transAxes, fontsize=12)
    #    ax2.text(.02, .97, 'B', ha='left', va='top', fontweight='bold', transform=ax2.transAxes, fontsize=12)
        if savefig:
            plt.tight_layout()
            plt.subplots_adjust(wspace=.02)
            plt.savefig(os.path.join(data_dir, 'figure', 'compared_with_farinotti_box_plot.pdf'), 
                        dpi=300, bbox_inches='tight')


def plot_total_ice_volume_compare(ax=None):
    
    def fillna(df, xcol, ycol, func=None, inplace=False):
        if not inplace:
            df = df.copy()
        if df[ycol].isna().sum() < 1:
            return df
        if not func:
            func = lambda x, a, b: a * x **b
        infdf = df[~df[ycol].isna()]
        nadf = df[df[ycol].isna()]
        popt, _ = curve_fit(func, infdf[xcol].values, infdf[ycol].values)
        func2 = lambda x: popt[0] * x ** popt[1]
        for id_, x in zip(nadf.index.values, nadf[xcol].values):
            df.loc[id_][ycol] = func2(x)
        return df
        
    glab_ggidf = pd.read_csv(glob.glob(os.path.join(cluster_dir, 
                                                    'tienshan_ice_volume', 
                                                    'Tienshan_glc_volume_uncertainty',
                                                    'glab_ggi_SRTM_opt*.csv'))[0])
    glab_rgidf = pd.read_csv(glob.glob(os.path.join(cluster_dir, 
                                                    'tienshan_ice_volume', 
                                                    'Tienshan_glc_volume_uncertainty',
                                                    'glab_rgi_SRTM_opt*.csv'))[0])
    oggm_rgidf = pd.read_csv(glob.glob(os.path.join(cluster_dir, 
                                                    'tienshan_ice_volume',
                                                    'Tienshan_glc_volume_uncertainty',
                                                    'oggm_rgi_SRTM_opt*.csv'))[0])
    oggm_ggidf = pd.read_csv(glob.glob(os.path.join(cluster_dir, 
                                                    'tienshan_ice_volume', 
                                                    'Tienshan_glc_volume_uncertainty',
                                                    'oggm_ggi_SRTM_opt*.csv'))[0])
    rgidf = gpd.read_file(os.path.join(data_dir, 'rgi'))

    fr_model_types = ['0', '1', '2', '3', '4']
    fr_vol_list = []
    for model_type in fr_model_types:
        fpath = os.path.join(root_dir, 'hma_vol', 'farinotti_thick', 'hma_glc_vol',
                             f'located_model{model_type}_km3.csv')
        hma_vol_df = pd.read_csv(fpath)
        hma_vol_df.index = hma_vol_df.rgi_id
        df = hma_vol_df.reindex(rgidf.RGIId.values)
        fr_vol_list.append(df[[f'model_{model_type}_km3']])
    fr_vol_df = pd.concat(fr_vol_list, axis=1)
    assert np.all(fr_vol_df.index == rgidf.RGIId)
    fr_vol_df['Area_km2'] = rgidf.Area.values
    f1 = lambda x, a, b: a * x ** b
    fr_vol_fix_df = fr_vol_df.copy()
    for column in fr_vol_df.columns:
        fr_vol_fix_df = fillna(fr_vol_fix_df, 'Area_km2', column)
            
    with open(os.path.join(cluster_dir, 'in_china_list.pkl'), 'rb') as f:
        in_ch = pickle.load(f)
    
    if not ax:
        fig, ax = plt.subplots(figsize=(6, 4))
    xlocs = np.arange(0, 8)    
#    colors = ['dodgerblue', 'firebrick', 'skyblue', 'lightcoral',
#              'darkorange', 'goldenrod', 'gold', 'burlywood', 'khaki']
    colors = ['dodgerblue', 'firebrick', 'skyblue', 'lightcoral', 'lightgrey',
              'lightgrey', 'lightgrey', 'lightgrey']
    
    i = 0
    width = .5
    for voldf in [glab_ggidf, oggm_ggidf, glab_rgidf, oggm_rgidf]:
        color, xloc = colors[i], xlocs[i]
        if i < 2:
            in_df = voldf[in_ch['ggi_in_China']]
            out_df = voldf[~np.array(in_ch['ggi_in_China'])]
        else:
            in_df = voldf[in_ch['rgi_in_China']]
            out_df = voldf[~np.array(in_ch['rgi_in_China'])]
        i += 1
        for df in [in_df, out_df]:
            fillna(df, 'rgi_area_km2', 'inv_volume_km3', inplace=True)

        ax.bar(xloc, in_df.inv_volume_km3.sum(), color=color, bottom=out_df.inv_volume_km3.sum(),
               hatch='//', ec='k', lw=1., width=width)
        ax.bar(xloc, out_df.inv_volume_km3.sum(), hatch='xx', color=color, ec='k', lw=1., width=width)
        ax.text(xloc, out_df.inv_volume_km3.sum()+in_df.inv_volume_km3.sum()+30,
                '{:.2f}'.format(out_df.inv_volume_km3.sum()/(in_df.inv_volume_km3.sum()+out_df.inv_volume_km3.sum())), 
                ha='center', va='center',fontweight='bold', fontsize=8)
        
    for column in fr_vol_df.columns[:-1]:
        color, xloc = colors[i], xlocs[i]
        i += 1
        voldf = fr_vol_df[[column, 'Area_km2']]
        in_df = voldf[in_ch['rgi_in_China']]
        out_df = voldf[~np.array(in_ch['rgi_in_China'])]        
        for df in [in_df, out_df]:
            fillna(df, 'Area_km2', column)
        ax.bar(xloc, in_df[column].sum(), bottom=out_df[column].sum(),
               color=color, hatch='//', ec='k', lw=1., width=width)
        ax.bar(xloc, out_df[column].sum(),hatch='xx', color=color, ec='k', lw=1., width=width)
        ax.text(xloc, out_df[column].sum()+in_df[column].sum()+30, 
                '{:.2f}'.format(out_df[column].sum()/(in_df[column].sum()+out_df[column].sum())),
                ha='center', va='center', fontweight='bold', fontsize=8)
    ax.bar(0, 0, 0, color='white', hatch='//', label='In China', ec='k')
    ax.bar(0, 0, 0, color='white', hatch='xx', label='Out of China', ec='k')
    ax.set_yticks(range(0, 1300, 300)), ax.set_ylim(0, 1199)
    ax.set_ylabel('Ice volume ($\mathregular{km^3}$)')
    ax.set_xticks(xlocs)
    ax.set_xticklabels(['BS/G', 'MC/G', 'BS/R', 'MC/R', 'F1/R',
                        'F2/R', 'F3/R', 'F4/R'])
    ax.set_xlabel('Model/Glacier inventory')
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.text(.02, .97, 'C', ha='left', va='top', fontweight='bold', transform=ax.transAxes, fontsize=12)
    ax.legend(labelspacing=.05, ncol=2, loc=1)


plot_compare_boxplot(plot_glc='specific', savefig=True, err='normalized')

plot_total_ice_volume_compare()
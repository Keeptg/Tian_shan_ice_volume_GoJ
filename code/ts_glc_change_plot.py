#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 08:53:01 2020

@author: keeptg
"""

import os, glob
import numpy as np
import pandas as pd
import xarray as xr

from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt

from path_config import *


def status_vol_area(gi_type, rcpid, cmip, rgiid=None, data_file=None, model_type=None):

    dir_ = os.path.join(root_dir, 'Data', 'cluster_output')
    if data_file is None:
        data_file = 'glacier_change_in_21st'
    ggi_rcp26_paths = glob.glob(os.path.join(dir_, 'tienshan_glacier_change',
                            data_file, f'{gi_type}_{cmip}',
                            f'*{rcpid}.nc'))
    gcm_name_list = [os.path.split(path)[-1].split('_')[2] for path in ggi_rcp26_paths]

    ggi_rcp26_vol_df = pd.DataFrame(columns=gcm_name_list, 
                                    index=range(2018, 2101))
    ggi_rcp26_area_df = pd.DataFrame(columns=gcm_name_list, 
                                     index=range(2018, 2101))
    glc_num_df = pd.DataFrame(columns=gcm_name_list, 
                              index=range(2018, 2101))
    print(ggi_rcp26_vol_df.isna().sum())

    for path in ggi_rcp26_paths:
        name = os.path.split(path)[-1].split('_')[2]
        compil_ds = xr.open_dataset(path)
        if rgiid is not None:
            compil_ds = compil_ds.sel(rgi_id=rgiid)
        time = compil_ds.time.values
        sum_compil = compil_ds.sum(dim='rgi_id')
        volume = sum_compil.volume.values * 1e-9
        area = sum_compil.area.values * 1e-6
        glc_num = (compil_ds.area>1e4).sum(dim='rgi_id').values
        ggi_rcp26_vol_df.loc[time, name] = volume
        ggi_rcp26_area_df.loc[time, name] = area
        glc_num_df.loc[time, name] = glc_num

    return ggi_rcp26_vol_df, ggi_rcp26_area_df, glc_num_df


def plot_glacier_change(ax, glc_chg_dfs, pre_glc_chg_df=None, time_range=(2018, 2100), normin=None, yass_inter=None,
                        show_volume=False):
    # colorlist1 = ['lightskyblue', 'lemonchiffon', 'mistyrose']
    colorlist1 = ['lightskyblue', 'none', 'lightcoral']
    colorlist2 = ['royalblue', 'blueviolet', 'firebrick']
    # hatchs = ['////', 'none', '\\\\\\\\']
    labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']


    if type(pre_glc_chg_df) is not type(None):
        norm_vol = pre_glc_chg_df.loc[normin]
        ax.plot(pre_glc_chg_df.index[1:], pre_glc_chg_df.iloc[1:].sum(axis=1)/norm_vol,
                color='k', label='Precent')

    output = []
    for i, df in enumerate(glc_chg_dfs):
        # mins = df.min(axis=1)
        # maxs = df.max(axis=1)
        mean = df.mean(axis=1)
        norm_vol = mean.loc[normin]
        mean /= norm_vol
        min_mass = mean.iloc[-1]
        std = (df.std(axis=1)/norm_vol).iloc[-1]
        output.append('{:.2%}+-{:.2%}'.format(min_mass, std))
        mins = mean - df.std(axis=1) / norm_vol
        maxs = mean + df.std(axis=1) / norm_vol
        diff = mean - .5
        v = diff.abs().min()
        time = diff[diff.abs()==v].index.values[0]
        v05 = mean[time]
        
        if i != 1:
        # ax.fill_between(df.index, mins, maxs, where=maxs>mins, lw=0.,
        #                 facecolor='none', interpolate=True, hatch=hatchs[i], ec=colorlist1[i])
            ax.fill_between(df.index, mins, maxs, where=maxs>mins, lw=1.,
                            color=colorlist2[i], interpolate=True, alpha=.1)
        ax.plot(mean.index, mean, color=colorlist2[i], label=labels[i])
        if i != 1 and v < 0.01:
            ax.plot([time, time], [v05, 0], color=colorlist2[i], ls=(0, (3, 2)), lw=.7)
            (ha, t) = ('right', -.05) if i > 1 else ('left', 1.5)
            ax.text(time+t, v05/2, f'{time}', color=colorlist2[i], fontsize=9, rotation=90,
                    va='center', ha=ha)
    ax.plot(time_range, (.5, .5), color='dimgrey', ls=(0, (3, 2)), lw=.7)
    ax.set_xlim(*time_range), ax.set_ylim(0, 1)
    ax.set_xticks([2020, 2050, 2080]), ax.set_yticks(np.arange(0.0, 1.01, 0.5))
    ax.set_yticklabels([0, 0.5, 1], rotation=90, va='center')
    ax.minorticks_on()
    ax.tick_params('both', length=4, which='major')
    ax.tick_params('both', length=1.5, which='minor')
    if show_volume:
        axt = ax.twinx()
        axt.set_ylim((0, norm_vol*1.21))
        axt.set_ylabel('Mass ($\mathregular{km^3}$)')
    if yass_inter:
        axt.set_yticks(np.arange(0, 1.21*norm_vol+1, yass_inter).astype(int))

    return output


def plot_delta_mass_change(ax, glc_chg_dfs, time_range=(2018, 2100), normin=None):
    # colorlist1 = ['lightskyblue', 'lemonchiffon', 'mistyrose']
    colorlist1 = ['lightskyblue', 'none', 'lightcoral']
    colorlist2 = ['royalblue', 'blueviolet', 'firebrick']
    # hatchs = ['////', 'none', '\\\\\\\\']
    labels = ['RCP2.6', 'RCP4.5', 'RCP8.5']

    for i, df in enumerate(glc_chg_dfs):
        # mins = df.min(axis=1)
        # maxs = df.max(axis=1)
        df_diff = pd.DataFrame(columns=df.columns, index=df.index[1:], data=df.iloc[:-1].values-df.iloc[1:].values)
        df_diff_5y = df_diff.rolling(1).mean()
        df_diff_5y.dropna(axis=0, how='all', inplace=True)
        mean = df_diff_5y.mean(axis=1)
        min_mass = mean.iloc[-1]
        std = (df_diff_5y.std(axis=1)).iloc[-1]
        mins = mean - df_diff_5y.std(axis=1)
        maxs = mean + df_diff_5y.std(axis=1)

        if i != 1:
            # ax.fill_between(df.index, mins, maxs, where=maxs>mins, lw=0.,
            #                 facecolor='none', interpolate=True, hatch=hatchs[i], ec=colorlist1[i])
            ax.fill_between(df_diff_5y.index, mins, maxs, where=maxs > mins, lw=1.,
                            color=colorlist2[i], interpolate=True, alpha=.1)
        ax.plot(mean.index, mean, color=colorlist2[i], label=labels[i])
    ax.plot(time_range, (.5, .5), color='dimgrey', ls=(0, (3, 2)), lw=.7)
    ax.set_xlim(*time_range)
    ax.set_xticks([2020, 2050, 2080])
    ax.set_ylabel('Mass (rel. 2015)')
    ax.legend(loc='lower left', labelspacing=.1)


dir_ = os.path.join(root_dir, 'Data', 'cluster_output')
cmip = 'cmip5'
# -----------------------------------------------------------------------------
china = gpd.read_file(os.path.join(dir_, 'china-shapefiles-master', 
                                    'simplied_china_country.shp'))
oggm_rgi1 = pd.read_csv(os.path.join(dir_,
                                     'tienshan_ice_volume',
                                     'Tienshan_glc_volume_uncertainty',
                                     'oggm_rgi_SRTM_opt(2.68).csv'))
#                                    'glacier_statistics_oggm_rgi.csv'))

oggm_gam1 = pd.read_csv(os.path.join(dir_,
                                     'tienshan_ice_volume',
                                     'Tienshan_glc_volume_uncertainty',
                                     'oggm_ggi_SRTM_opt(2.68).csv'))
#                                     'glacier_statistics_oggm_gam.csv'))
rgi_inchina = [china.geometry.contains(Point(x, y)).values[0] for
                x, y in zip(oggm_rgi1.cenlon, oggm_rgi1.cenlat)]
ggi_inchina = [china.geometry.contains(Point(x, y)).values[0] for
                x, y in zip(oggm_gam1.cenlon, oggm_gam1.cenlat)]
rgiid_in = oggm_rgi1.rgi_id.loc[rgi_inchina].values
ggiid_in = oggm_gam1.rgi_id.loc[ggi_inchina].values
rgiid_out = oggm_rgi1.rgi_id.loc[~np.array(rgi_inchina)].values
ggiid_out = oggm_gam1.rgi_id.loc[~np.array(ggi_inchina)].values

path = os.path.join(dir_, 'tienshan_glacier_change', 'glacier_change_in_21st',
                    f'ggi_{cmip}')
oggm_pre_area_ggi = pd.read_csv(os.path.join(path, 'stat_area.csv'))
oggm_pre_vol_ggi = pd.read_csv(os.path.join(path, 'stat_vol.csv'))
oggm_pre_vol_ggi = oggm_pre_vol_ggi.set_index('Unnamed: 0') * 1e-9
oggm_pre_area_ggi = oggm_pre_area_ggi.set_index('Unnamed: 0') * 1e-6

path = os.path.join(dir_, 'tienshan_glacier_change', 'glacier_change_in_21st',
                    f'rgi_{cmip}')
#oggm_pre_area_rgi = pd.read_csv(os.path.join(path, 'stat_area.csv'))
oggm_pre_vol_rgi = pd.read_csv(os.path.join(path, 'stat_vol.csv'))
oggm_pre_vol_rgi = oggm_pre_vol_rgi.set_index('Unnamed: 0') * 1e-9
#oggm_pre_area_rgi = oggm_pre_area_rgi.set_index('Unnamed: 0') * 1e-6

oggm_ggi_rcp26_vol_df, oggm_ggi_rcp26_area_df, oggm_ggi_rcp26_num_df = \
    status_vol_area('ggi', 'rcp26', cmip)
oggm_ggi_rcp45_vol_df, oggm_ggi_rcp45_area_df, oggm_ggi_rcp45_num_df = \
    status_vol_area('ggi', 'rcp45', cmip)
oggm_ggi_rcp85_vol_df, oggm_ggi_rcp85_area_df, oggm_ggi_rcp85_num_df = \
    status_vol_area('ggi', 'rcp85', cmip)
oggm_rgi_rcp26_vol_df, oggm_rgi_rcp26_area_df, oggm_rgi_rcp26_num_df = \
    status_vol_area('rgi', 'rcp26', cmip)
oggm_rgi_rcp45_vol_df, oggm_rgi_rcp45_area_df, oggm_rgi_rcp45_num_df = \
    status_vol_area('rgi', 'rcp45', cmip)
oggm_rgi_rcp85_vol_df, oggm_rgi_rcp85_area_df, oggm_rgi_rcp85_num_df = \
    status_vol_area('rgi', 'rcp85', cmip)

ggi_rcp26_vol_df_in, ggi_rcp26_area_df_in, ggi_rcp26_num_df_in = \
    status_vol_area('ggi', 'rcp26', cmip, ggiid_in)
ggi_rcp45_vol_df_in, ggi_rcp45_area_df_in, ggi_rcp45_num_df_in = \
    status_vol_area('ggi', 'rcp45', cmip, ggiid_in)
ggi_rcp85_vol_df_in, ggi_rcp85_area_df_in, ggi_rcp85_num_df_in = \
    status_vol_area('ggi', 'rcp85', cmip, ggiid_in)
rgi_rcp26_vol_df_in, rgi_rcp26_area_df_in, rgi_rcp26_num_df_in = \
    status_vol_area('rgi', 'rcp26', cmip, rgiid_in)
rgi_rcp45_vol_df_in, rgi_rcp45_area_df_in, rgi_rcp45_num_df_in = \
    status_vol_area('rgi', 'rcp45', cmip, rgiid_in)
rgi_rcp85_vol_df_in, rgi_rcp85_area_df_in, rgi_rcp85_num_df_in = \
    status_vol_area('rgi', 'rcp85', cmip, rgiid_in)

# out china data prepare
ggi_rcp26_vol_df_out, ggi_rcp26_area_df_out, ggi_rcp26_num_df_out = \
    status_vol_area('ggi', 'rcp26', cmip, ggiid_out)
ggi_rcp45_vol_df_out, ggi_rcp45_area_df_out, ggi_rcp45_num_df_out = \
    status_vol_area('ggi', 'rcp45', cmip, ggiid_out)
ggi_rcp85_vol_df_out, ggi_rcp85_area_df_out, ggi_rcp85_num_df_out = \
    status_vol_area('ggi', 'rcp85', cmip, ggiid_out)
rgi_rcp26_vol_df_out, rgi_rcp26_area_df_out, rgi_rcp26_num_df_out = \
    status_vol_area('rgi', 'rcp26', cmip, rgiid_out)
rgi_rcp45_vol_df_out, rgi_rcp45_area_df_out, rgi_rcp45_num_df_out = \
    status_vol_area('rgi', 'rcp45', cmip, rgiid_out)
rgi_rcp85_vol_df_out, rgi_rcp85_area_df_out, rgi_rcp85_num_df_out = \
    status_vol_area('rgi', 'rcp85', cmip, rgiid_out)


def plot_fig1():

    fig, axs = plt.subplots(3, 2, figsize=(8.5, 6), sharex=True)
    norm_time = 2018
    min_mass = []
    # ------
    min_mass.append(plot_glacier_change(axs[0, 0], [oggm_ggi_rcp26_vol_df, oggm_ggi_rcp45_vol_df,
                                    oggm_ggi_rcp85_vol_df], 
                        normin=norm_time, yass_inter=150))
    
    min_mass.append(plot_glacier_change(axs[0, 1], [oggm_rgi_rcp26_vol_df, oggm_rgi_rcp45_vol_df,
                                    oggm_rgi_rcp85_vol_df], 
                       normin=norm_time, yass_inter=150))
    
    # ------
    
    min_mass.append(plot_glacier_change(axs[1, 0], [ggi_rcp26_vol_df_in, ggi_rcp45_vol_df_in,
                                    ggi_rcp85_vol_df_in],
                        normin=norm_time, yass_inter=100))
    
    min_mass.append(plot_glacier_change(axs[1, 1], [rgi_rcp26_vol_df_in, rgi_rcp45_vol_df_in,
                                    rgi_rcp85_vol_df_in], 
                        normin=norm_time, yass_inter=100))
    
    # ------
    
    min_mass.append(plot_glacier_change(axs[2, 0], [ggi_rcp26_vol_df_out, ggi_rcp45_vol_df_out,
                                    ggi_rcp85_vol_df_out], 
                         normin=norm_time, yass_inter=100))
    
    min_mass.append(plot_glacier_change(axs[2, 1], [rgi_rcp26_vol_df_out, rgi_rcp45_vol_df_out,
                                    rgi_rcp85_vol_df_out], 
                         normin=norm_time, yass_inter=100))
    
    for ax, alpha in zip(axs.flatten(), ['A', 'B', 'C', 'D', 'E', 'F']):
        ax.text(.99, .98, alpha, fontweight='bold', ha='right', va='top', 
                transform=ax.transAxes)
    
    # -------------------------------------------------------------------------
    
    # for i in [0, 1, 2]:
    #     axs[i, 0].set_yticks(np.arange(0.1, 1.3, .2))
    #     axs[i, 0].set_ylim(0, 1.2)
    #     axs[i, 0].set_ylabel(f'Mass (rel. to {norm_time})')
    for i in [0, 1]:
        # axs[2, i].set_xticks([2020, 2050, 2080])
        axs[2, i].set_xlabel('Year')
        # axs[2, i].set_xlim(2000, 2100)
    
    # fig.text(.05, .5, f'Mass (rel. to {norm_time})', ha='left', va='center', rotation=90)
    plt.tight_layout(h_pad=.02, w_pad=1)
    
    fig.savefig(os.path.join(data_dir, 'figure', 'glacier_change.pdf'), pdi=300,
                bbox_inches='tight')

def get_rouce2020_vol(region=None):
    import pickle
    r_path = os.path.join(root_dir, 'Data', 'Rouce2020')
    if not region:
        outpath = os.path.join(r_path, 'vol_in_tienshan.pkl')
        rgiid_tienshan = oggm_rgi1.rgi_id.values
    elif region == 'in_china':
        outpath = os.path.join(r_path, 'vol_in_china_tienshan.pkl')
        rgiid_tienshan = rgiid_in
    elif region == 'out_china':
        outpath = os.path.join(r_path, 'vol_out_china_tienshan.pkl')
        rgiid_tienshan = rgiid_out
    else:
        raise ValueError("Wrong region!")
    
    if os.path.exists(outpath):
        with open(outpath, 'rb') as f:
            output = pickle.load(f)
    else:
        rouce_nc = glob.glob(os.path.join(root_dir, 'Data', 
                                          'Rouce2020', '*.nc'))
        import pickle
        vol_list = []
        rcp_list = []
        for path in rouce_nc:
            rcp = path.split('_')[7]
            rcp_list.append(rcp)
            ds = xr.open_dataset(path)
            rc_in_ts = [values in rgiid_tienshan for values in ds.RGIId.values]
            sub_ds = ds.sel(glac=rc_in_ts)
            vol = sub_ds.glac_volume_annual.sum(dim='glac')
            vol_list.append(vol)
        output = dict(zip(rcp_list, vol_list))
        with open(outpath, 'wb') as f:
            pickle.dump(output, f)
            
    return output


def glc_cha_with_oggm_glab_rouce():
    fig, axs = plt.subplots(2, 2, sharey=True)
    rouce20 = get_rouce2020_vol()
    # ax = axs[0]
    def sub_plot(ax, glc_cha_dfs, label, normin, legend=True, pre_glc_cha_df=None,
                 xlabel=True, ylabel=True):
        plot_glacier_change(ax, glc_cha_dfs, pre_glc_cha_df, normin=normin)
        rouce20 = get_rouce2020_vol()
        c_dic = {'rcp26': 'royalblue', 'rcp45': 'orange', 'rcp85': 'firebrick'}
        norm_rouce20 = rouce20['rcp26'].sel(year_plus1=normin).values
        for key in rouce20.keys():
            if key == 'rcp60':
                continue
            vol = rouce20[key]/norm_rouce20
            color = c_dic[key]
            time = vol.year_plus1.values
            ax.plot(time, vol, lw=2, color=color, ls=(0, (3, 2, )))
        ax.bar(2000, 0, 0, color=c_dic['rcp26'], label='rcp26')
        ax.bar(2000, 0, 0, color=c_dic['rcp45'], label='rcp45')
        ax.bar(2000, 0, 0, color=c_dic['rcp85'], label='rcp85')
        ax.plot((2000, 2000), (100, 100), color='dimgrey', label=label)
        ax.plot((2000, 2000), (100, 100), color='dimgrey', label='Rouce2020', 
                ls=(0, (3, 2,)))
        ax.set_xlim(2000, 2100)
        ax.set_ylim(0, 1.2)
        if legend:
            ax.legend()
        if xlabel:
            ax.set_xlabel('Years')
        if ylabel:
            ax.set_ylabel(f'Ice volume (rel. to {normin})')
            
    sub_plot(axs[0, 0], [oggm_rgi_rcp26_vol_df, oggm_rgi_rcp45_vol_df, 
                         oggm_rgi_rcp85_vol_df],
             normin=2018, label='OGGM(RGI)')
    
    sub_plot(axs[0, 1], [oggm_ggi_rcp26_vol_df, oggm_ggi_rcp45_vol_df, 
                         oggm_ggi_rcp85_vol_df],
             normin=2018, label='OGGM(GGI)', ylabel=False)
    
        
# glc_cha_with_oggm_glab_rouce()
plot_fig1()
glc_cha_with_oggm_glab_rouce()
        

default_oggm_rgi_rcp26_vol_df = status_vol_area('rgi', cmip, rcpid='rcp26', data_file='glacier_change')[0]
default_oggm_rgi_rcp45_vol_df = status_vol_area('rgi', cmip, rcpid='rcp45', data_file='glacier_change')[0]
default_oggm_rgi_rcp85_vol_df = status_vol_area('rgi', cmip, rcpid='rcp85', data_file='glacier_change')[0]

default_oggm_rgi_rcp26_vol_df_in = status_vol_area('rgi', cmip, rcpid='rcp26', data_file='glacier_change', rgiid=rgiid_in)[0]
default_oggm_rgi_rcp45_vol_df_in = status_vol_area('rgi', cmip, rcpid='rcp45', data_file='glacier_change', rgiid=rgiid_in)[0]
default_oggm_rgi_rcp85_vol_df_in = status_vol_area('rgi', cmip, rcpid='rcp85', data_file='glacier_change', rgiid=rgiid_in)[0]

default_oggm_rgi_rcp26_vol_df_out = status_vol_area('rgi', cmip, rcpid='rcp26', data_file='glacier_change', rgiid=rgiid_out)[0]
default_oggm_rgi_rcp45_vol_df_out = status_vol_area('rgi', cmip, rcpid='rcp45', data_file='glacier_change', rgiid=rgiid_out)[0]
default_oggm_rgi_rcp85_vol_df_out = status_vol_area('rgi', cmip, rcpid='rcp85', data_file='glacier_change', rgiid=rgiid_out)[0]

fig, ax = plt.subplots(3, 2, figsize=(4, 5.5), sharex=True, sharey=True)

plot_glacier_change(ax[2, 0], [oggm_rgi_rcp26_vol_df, oggm_rgi_rcp45_vol_df,
                               oggm_rgi_rcp85_vol_df], normin=2018)
plot_glacier_change(ax[2, 1], [oggm_ggi_rcp26_vol_df, oggm_ggi_rcp45_vol_df,
                                  oggm_ggi_rcp85_vol_df], normin=2018)
plot_glacier_change(ax[1, 0], [rgi_rcp26_vol_df_in, rgi_rcp45_vol_df_in,
                                  rgi_rcp85_vol_df_in], normin=2018)
plot_glacier_change(ax[1, 1], [ggi_rcp26_vol_df_in, ggi_rcp45_vol_df_in,
                                  ggi_rcp85_vol_df_in], normin=2018)
plot_glacier_change(ax[0, 0], [rgi_rcp26_vol_df_out, rgi_rcp45_vol_df_out,
                                  rgi_rcp85_vol_df_out], normin=2018)
plot_glacier_change(ax[0, 1], [ggi_rcp26_vol_df_out, ggi_rcp45_vol_df_out,
                                  ggi_rcp85_vol_df_out], normin=2018)

ax[0, 0].legend(loc='upper right', labelspacing=.1, handlelength=1, handletextpad=.2, frameon=False)
ax[1, 0].set_ylabel('Mass (rel. 2018)')
ax[2, 0].set_xlabel('Year'), ax[2, 1].set_xlabel('Year')
labels = ['A', 'B', 'C', 'D', 'E', 'F']
for i, a in zip(labels, ax.flatten()):
    a.text(.03, .03, i, weight='bold', transform=a.transAxes, ha='left', va='bottom', fontsize=10)

plt.tight_layout(h_pad=.1, w_pad=.1)
fig.savefig(os.path.join(data_dir, 'figure', 'glc_change_in_21st.pdf'),
            dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(3, 2, figsize=(4, 5.5), sharex='col', sharey='row')

plot_delta_mass_change(ax[2, 0], [oggm_rgi_rcp26_vol_df, oggm_rgi_rcp45_vol_df,
                               oggm_rgi_rcp85_vol_df])
plot_delta_mass_change(ax[2, 1], [oggm_ggi_rcp26_vol_df, oggm_ggi_rcp45_vol_df,
                               oggm_ggi_rcp85_vol_df])
plot_delta_mass_change(ax[1, 0], [rgi_rcp26_vol_df_in, rgi_rcp45_vol_df_in,
                               rgi_rcp85_vol_df_in])
plot_delta_mass_change(ax[1, 1], [ggi_rcp26_vol_df_in, ggi_rcp45_vol_df_in,
                               ggi_rcp85_vol_df_in])
plot_delta_mass_change(ax[0, 0], [rgi_rcp26_vol_df_out, rgi_rcp45_vol_df_out,
                               rgi_rcp85_vol_df_out])
plot_delta_mass_change(ax[0, 1], [ggi_rcp26_vol_df_out, ggi_rcp45_vol_df_out,
                               ggi_rcp85_vol_df_out])

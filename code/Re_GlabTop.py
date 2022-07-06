#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:05:55 2020

Model Name: Re-GlabTop

Main References:
    
    Linsbauer A, Paul F, Haeberli W (2012) Modeling glacier thickness 
        distribution and bed topography over entire mountain ranges with 
        GlabTop: Application of a fast and robust approach. 
        Journal of Geophysical Research: Earth Surface 117:n/a-n/a 
        doi:10.1029/2011jf002313
        
    Carrivick JL, Davies BJ, James WHM et al. (2016) Distributed ice thickness 
        and glacier volume in southern South America. Global and Planetary 
        Change 146:122-132 doi:10.1016/j.gloplacha.2016.09.010

@author: Fei Li

"""


import os
from os.path import join
import copy
import logging
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import geopandas as gpd
import rasterio
import salem
import netCDF4 as nc
import xarray as xr
from oggm import entity_task
try:
    import gdal
except:
    from osgeo import gdal
from oggm import cfg, utils, entity_task

log = logging.getLogger(__name__)

def calculate_slope(DEM, Slope=''):
    # TODO: I used gdal to calcuate DEM slope and get a tiff files. But this 
    # is not oggm's style. So I will try write the slope data to 
    # gridded_data.nc file and make it seems more like oggm
    """Calculate slope with dem.tif

    Parameters
    ----------
    DEM: the path and name of DEM
    Slope: the output path and name of Slope file,
        defult path is same with DEM

    Return
    ------
    ds: np.ndarray, the slope array
    """
    if Slope:
        slope_path = Slope
    else:
        slope_path = os.path.join(os.path.split(DEM)[0], 'slope.tif')
    gdal.DEMProcessing(slope_path, DEM, 'slope')
    with rasterio.open(slope_path) as rf:
        ds = rf.read(1)

    return ds


def inverse_distance_weight_interpo_1d(ls):
    """Interpolate list which has NaN

    Parameter
    ---------
    ls: np.darray

    Return
    ------
    new_ls, np.darray, interpolated array
    """

    n = np.array(np.where(np.isnan(ls)))[0]
    f = np.array(np.where(np.isfinite(ls)))[0]
    new_ls = copy.copy(ls)
    dis = []
    for a in n:
        dis = np.abs(f-np.full(np.shape(f), a))
        lam = 1 / dis
        slam = np.sum(lam)
        sls = np.sum(ls[np.isfinite(ls)]*lam)
        new_ls[a] = sls / slam

    return new_ls


@entity_task(log, writes=['inversion_input'])
def prepare_Glabtop_inputs(gdir):
    """Obtain the max/min elevation(max_h/min_h), area(catchment_area) , 
        slope and flowline length(length) of each individual catchment 
        and write them in 'inversion_input.plk'

    Parameters
    ----------
    gdir : :py:class:'oggm.GlacierDirectory'
        where to write the data
        
    suffix :str : if not '', the there should be a suppllment calculation 
        process. For now, the suffix = '1', if need.
    """

    # Give the minimum slope limitation (in degree)
    # TODO: maybe min_slope and max_slope should be put in cfg.PARAMS
    
    # min_slope = 7
    min_slope = cfg.PARAMS['min_slope']
    # min_slope = 5
    max_slope = 90

    fpath = gdir.get_filepath('gridded_data')

    geom = gdir.read_pickle('geometries')

    slope_path = os.path.join(os.path.split(gdir.get_filepath('dem'))[0],
                              'slope.tif')
    try:
        with rasterio.open(slope_path) as rf:
            slopes = rf.read(1)
    except:
        dem_path = gdir.get_filepath('dem')
        slopes = calculate_slope(dem_path)

    catchment_ids = geom['catchment_indices']
    inversion_inputs = gdir.read_pickle('inversion_input')
    fls = gdir.read_pickle('inversion_flowlines')
    max_h = []
    min_h = []
    towrite = []

    with utils.ncDataset(fpath) as nc:
        topo = nc.variables['topo_smoothed'][:]
        glacier_mask = nc.variables['glacier_mask'][:]
    slopes = np.where(glacier_mask==1, slopes, np.nan)

    with utils.ncDataset(fpath, 'a') as nc:
        vn = 'slope'
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            v = nc.createVariable(vn, 'f4', ('y', 'x', ), zlib=True)
        v.units = 'Degree'
        v.long_name = 'Slope of Ice Surface'
        v[:] = slopes

    slope = []
    for fl in fls:
        x, y = utils.tuple2int(fl.line.xy)
        slope_smooth = []
        for xi, yi in zip(x, y):
            s3 = np.nanmean(slopes[yi-1:yi+2, xi-1:xi+2])
            s5 = np.nanmean(slopes[yi-2:yi+3, xi-2:xi+3])
            s7 = np.nanmean(slopes[yi-3:yi+4, xi-3:xi+4])
            if s3 > 20:
                slope_smooth.append(s3)
            elif s5>5 and s5 <= 20:
                slope_smooth.append(s5)
            elif s7 < 5:
                slope_smooth.append(s7)
            else:
                slope_smooth.append(min_slope)
        slope_smooth = np.clip(slope_smooth, min_slope, max_slope)

        if np.any(np.isnan(slope_smooth)):
            slope_smooth = np.array(slope_smooth)
            slope_smooth = inverse_distance_weight_interpo_1d(slope_smooth)
            if slope_smooth < min_slope:
                slope_smooth == min_slope
        slope.append(np.deg2rad(slope_smooth))

    for pet in catchment_ids:
        max_h.append(np.max(topo[tuple(pet.T)]))
        min_h.append(np.min(topo[tuple(pet.T)]))

    for pet, inputs in enumerate(inversion_inputs):
        length = fls[pet].nx * fls[pet].dx * gdir.grid.dx
        catchment_area_fls = np.sum(fls[pet].widths) * fls[pet].dx * \
            gdir.grid.dx**2
        catchment_area_geo = len(geom['catchment_indices'][pet]) * \
            gdir.grid.dx**2
        inputs['slope'] = slope[pet]
        inputs['length'] = length
        inputs['catchment_area_fls'] = catchment_area_fls
        inputs['catchment_area_geo'] = catchment_area_geo
        inputs['max_h'] = max_h[pet]
        inputs['min_h'] = min_h[pet]
        towrite.append(inputs)

    gdir.write_pickle(towrite, 'inversion_input')

    return


def calculate_tal1(gdir, c=0.5, b=159.8, a=-43.5):
    
    """
    Calculate base stress. Based on the empirical relationship used by 
        Linsbauer A et al. (2012) but not same. The original empirical 
        relationship was parameterized in there.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
         the directory to process.
    max_h : float: the maximal elevation of goal calculation region
    min_h : float: he minimal elevation of goal calculation region
    a, b, c : float: parameterized parameters
        defult: None, the value of the original empirical relationship
    
    return
    ------
    tal : float: base stress values in the unit of Kpa
    
    """
    
    if c == None:
        c = 0.5
    if b == None:
        b = 159.8
    if a == None:
        a = -43.5
        
    glacier_stat = utils.glacier_statistics(gdir)
    max_h = glacier_stat['dem_max_elev']
    min_h = glacier_stat['dem_min_elev']
    deltH = (max_h - min_h)/1000
    tal = (c + b * deltH + a * deltH**2) * 1e3
    if (c==0.5 and b==159.8 and a==-43.5):
        if float(deltH) > 1.6:
            tal = 150 * 1e3
    else:
        if -b / (2 * a) < deltH:
            tal = c - b**2 / (4 * a)
    return tal


def calculate_tal2(gdir, a=None, b=None):
    """
    Calculate base stress. Based on the empirical relationship used by 
        Carrivick et al.(2016) but not same. The original empirical 
        relationship was parameterized in there.
        
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
         the directory to process.
    a, b : float: parameterized parameters
        defult: None, the value of the original empirical relationship
    
    return
    ------
    tal : float: base stress values in the unit of Kpa

    """

    tal = 0
    cls = gdir.read_pickle('inversion_input')
    if a is None:
        a = 2.7
    if b is None:
        b = 0.15
    for cl in cls:
        area = cl['dx'] * cl['width']
        slope = cl['slope_angle']
        cos_a = np.cos(np.arctan(slope))
        tal += a * 10e4 * np.nansum((area/cos_a)**b)
    tal /= 1000

    return tal


@entity_task(log, writes=['inversion_output'])
def base_stress_inversion(gdir, talmethod='1', sf_method=0.8, n=None, min_slope=None,
                          write=True, filesuffix='', calib_tal=None, use_pixel_slope=True,
                          **tal_param_dict):

    """
    Computes the glacier thickness alone the flowlines

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
         the directory to process.

    talmethod : str: '1' or '2' optional, choose the calculateion method of 
        base stress 'tal'
        '1' for 'calculate_tal1' function and '2' for calculate_tal2' function
        defult: '2'

    sf_method : float or None:
        choose the calculation method of shape factor(sf)
        if None, sf is the function of tal and glacier width
        if float type, sf = sf_method, defult 0.8

    n : None or float:
        parameter of calculation sf, defult 0.5

    write : bool
        default behavior is to compute the thickness and write the
        results in the pickle. Set to False in order to spare time
        during calibration.

    filesuffix : str
        add a suffix to the output file

    calib_tal : float
        used for calibrate tal
        
    tar_param_dict: dict
        a, b, c (if talmethod=='1') : None or float:
            parameters of calculate_tal1 function, 
            only used when tal_method='1'
        a, b (if talmethod=='2'): None or float:
            parameters of calculate_tal2 function, 
            only used when tal_method='2'

    """

    rho = cfg.PARAMS['ice_density']
    g = cfg.G

# TODO: Maybe min_sf parameter should be put in cfg.PARAMS
    cls = gdir.read_pickle('inversion_input')
    if not min_slope:
        interq = 5
        slopes = []
        for cl in cls:
            slopes = np.append(slopes, cl['slope'])
        min_slope = np.rad2deg(np.percentile(slopes, interq))
        if min_slope < cfg.PARAMS['min_slope']:
            min_slope = cfg.PARAMS['min_slope']
        elif min_slope > 5:
            min_slope = 5
    elif type(min_slope) is str:
        min_slope = float(min_slope)

    if talmethod == '1':
        tal = calculate_tal1(gdir, **tal_param_dict)
        
    elif talmethod == '2':
        tal = calculate_tal2(gdir=gdir, **tal_param_dict)
    else:
        raise ValueError(f"No talmethod is {talmethod}")
    if calib_tal:
        tal *= calib_tal

    if n is None:
        n = 0.5
    for cl in cls:
        w = cl['width']
        slope = cl['slope']
        if use_pixel_slope:
            slope = np.clip(slope, np.deg2rad(min_slope), np.pi/2)
        else:
            slope = np.arctan(cl['slope_angle'])
        if sf_method is None:
            sf = 1 - 2 * tal / (w * n * rho * g * np.sin(slope))
            sf = np.clip(sf, .4, 1)
        else:
            sf = float(sf_method)
            if sf < 0.3:
                sf = 0.3
            elif sf > 1:
                sf = 1
        out_thick = tal / (rho * g * np.sin(slope) * sf)

        # volume
        fac = np.where(cl['is_rectangular'], 1, 2./3.)
        volume = fac * out_thick * w * cl['dx']
        if write:
            cl['thick'] = out_thick
            cl['volume'] = volume
            cl['shape_factor'] = sf
    if write:
        gdir.write_pickle(cls, 'inversion_output', filesuffix=filesuffix)


def proj(x, y, gdir):
    """Project x, y from geographical coordinates system to grid's
    plane_coordinate system

    Parameters
    ----------
    x: longitude value
    y: lantitude value
    gdir : :py:class:`oggm.GlacierDirectory`
         the directory to process.

    Return
    ------
    tuple
    """

    grid = gdir.grid
    return grid.transform(x, y, crs=salem.wgs84)


def delete_variable_from_netcdf(input_nc, del_variable_list):
    """Delete variables from the input netcdf file
    This function is recoded from:
    'https://stackoverflow.com/questions/15141563/', Rich Signell's answer

    Parameters
    ----------
    input_nc: the netcdf file and path
    del_variable_list: a list which contains the variable name you want delete
    """

    output_path = os.path.split(input_nc)[0]
    with nc.Dataset(input_nc) as src, nc.Dataset(
            os.path.join(output_path, "out.nc"), "w") as dst:
        # copy global attributes all at once via dictionary
        dst.setncatts(src.__dict__)
        # copy dimensions
        for name, dimension in src.dimensions.items():
            dst.createDimension(
                name, (len(dimension) if not dimension.isunlimited() else None))
        # copy all file data except for the excluded
        for name, variable in src.variables.items():
            if name not in del_variable_list:
                dst.createVariable(name, variable.datatype,
                                        variable.dimensions)
                dst[name][:] = src[name][:]
                # copy variable attributes all at once via dictionary
                dst[name].setncatts(src[name].__dict__)
    os.remove(input_nc)
    os.rename(os.path.join(output_path, "out.nc"), input_nc)


def reproject_shpfile_to_gdir_crs(grid, gpdf):
    """Reproject a shpfile to the gdir's crs

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
         the directory to process.
    gpdf: geopandas.GeoDataFrame.

    Return
    ------
    gpdf: geopandas.GeoDataFrame.
    """

    crs = grid.center_grid.proj
    salem.transform_geopandas(gpdf, to_crs=crs, inplace=True)

    return gpdf


def locate_point_in_grid(x, y, gdir):
    """Locate the location of a point with lon-lat coordination in grid

    Parameters
    ----------
    x : float: longitude
    y : float: latitude
    gdir : :py:class:`oggm.GlacierDirectory`
         the directory to process.

    Return
    ------
    (x, y): (int, int), the location in gdir's grid
    """

    dx, dy = gdir.grid.dx, gdir.grid.dy
    x0, y0 = gdir.grid.x0, gdir.grid.y0
    x = int((x - x0 + dx / 2) / abs(dx))
    y = int((y0 - y - dy / 2) / abs(dy))

    return x, y


#def _gridded_measured_thickness_point(grid, mea_thick_df, mask_df=None,
#                                      thick_col=None, method='mean'):
#
#    srs = grid.proj.srs
#    thick_df = mea_thick_df.copy().to_crs(srs)
#    if mask_df is not None:
#        mask_df = mask_df.copy().to_crs(srs)
#        thick_df = gpd.clip(thick_df, mask_df)
#
#    if not thick_col:
#        thick_col = 'thickness'
#
#    nx, ny = grid.nx, grid.ny
#    dx, dy = grid.dx, grid.dy
#    x0, y0 = grid.x0, grid.y0
#    thick = np.full((ny, nx), np.nan)
#    renum = np.ones((ny, nx))
#    # thick_point = reproject_shpfile_to_gdir_crs(grid, mea_thick_df)
#
#    for lon, lat, t in zip(thick_df.geometry.x, thick_df.geometry.y,
#                           thick_df[thick_col]):
#        # lon, lat = thick_df.iloc[i].geometry.xy
#        x = int((lon - x0 + dx / 2) / abs(dx))
#        y = int((y0 - lat - dy / 2) / abs(dy))
#        t = float(t)
#        try:
#            if np.isfinite(thick[y, x]):
#                renum[y, x] += 1
#                thick[y, x] += t
#            else:
#                thick[y, x] = t
#        except IndexError:
#            continue
#
#    if method == 'mean':
#        thick /= renum
#    elif method =='sum':
#        pass
#    else:
#        raise ValueError(f"No method named {method}!")
#
#    return thick
def _gridded_measured_thickness_point(grid, mea_thick_df, mask_df=None,
                                      thick_col=None, method='mean'):

    if thick_col is None:
        thick_col = 'thickness'
    if mask_df is not None:
        mask_df = mask_df.to_crs(mea_thick_df.crs)
        mea_thick_df = gpd.clip(mea_thick_df, mask_df)
    xs, ys = mea_thick_df.geometry.x, mea_thick_df.geometry.y
    xt, yt = grid.transform(xs, ys, nearest=True, crs=mea_thick_df.crs, maskout=True)
    mask = xt.mask
    xt = xt.data[~xt.mask]
    yt = yt.data[~yt.mask]

    mea_thick_df = copy.copy(mea_thick_df)
    mea_thick_df = mea_thick_df[~mask]
    mea_thick_df['loc'] = [(y, x) for y, x in zip(yt, xt)]
    _mea_thick_df = mea_thick_df.groupby('loc').mean()
    thick = np.full((grid.ny, grid.nx), np.nan)
    for i in range(len(_mea_thick_df)):
        df = _mea_thick_df.iloc[[i]]
        loc = df.index.values[0]
        thick[loc[0], loc[1]] = df[thick_col].values[0]

    return thick


@entity_task(log)
def gridded_measured_thickness_point(gdir, mea_thick_df=None, rewrite=False,
                                     thick_col=None, method='mean'):
    """Write measured thickness data to gridded_data.nc file

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
         the directory to process.
    mea_thick_dfk_df : GeoDataFrame: measured glacier thickness point data
    thick_col : str: point out the column name of thickness data
    
    """
    
    glacier_outline = gdir.read_shapefile('outlines')

    if not rewrite:
        with utils.ncDataset(gdir.get_filepath('gridded_data')) as ncf:
            if 'thick' in ncf.variables:
                thick = ncf['thick'][:]
                return thick

    grid = gdir.grid
    thick = _gridded_measured_thickness_point(grid, mea_thick_df, 
                  mask_df=glacier_outline, thick_col=thick_col, method='mean')
    
    del_list = ['thick']
    delete_variable_from_netcdf(gdir.get_filepath('gridded_data'), del_list)

    # Put thickness_point in grided_data's grid
    with utils.ncDataset(gdir.get_filepath('gridded_data'), mode='r+') as ncf:
        try:
            glacier_mask = ncf['glacier_mask'][:]
            thick = np.where(glacier_mask == 0, np.nan, thick)
        except IndexError:
            pass

        v = ncf.createVariable('thick', 'f', ('y', 'x', ))
        v.units = 'm'
        v.long_name = 'measured_thickness'
        v[:] = thick
        
    return thick


def extract_cal_mea_thickness(gdir, write=True, path=None):
    """Extract calculate thickness which is coincident with measured thickness
    grid

    Parameter
    ---------
    gdir : :py:class:`oggm.GlacierDirectory`
         the directory to process.
    write : bool: 
	 if write the result as a csv file to the specified path or not.
	 defult True, write
    path : None or str:
	 specify the path for the writen file
	 only useful when the 'write' keyword is True
	 defult None, the path of the glacier's working directory
    Returns
    -------
    calcu_thick : :py: pandas.Series
        calculated ice thickness series
    measure_thick : :py: pandas.Series
        measured ice thickness series
    """

    with xr.open_dataset(join(gdir.dir, 'gridded_data.nc')) as ncf:
        calcu_thick = ncf['distributed_thickness'].values
        measure_thick = ncf.thick.values
        s_elev = ncf.topo_smoothed.values
        try:
            cali_meas_thick = ncf.cali_meas_thick.values
        except AttributeError:
            cali_meas_thick = np.full(np.shape(measure_thick), np.nan)

    calcu_thick = calcu_thick[np.where(np.isfinite(measure_thick))]
    cali_meas_thick = cali_meas_thick[np.where(np.isfinite(measure_thick))]
    s_elev = s_elev[np.where(np.isfinite(measure_thick))]
    measure_thick = measure_thick[np.where(np.isfinite(measure_thick))]
    
    if write:
        if path is None:
            path = gdir.dir
        output = pd.DataFrame({'measure_thick': measure_thick,
                               'cali_meas_thick': cali_meas_thick,
                               'calcu_thick': calcu_thick,
                               'surface_elev': s_elev})
        output.to_csv(join(path, 'calculated_thickness_verify.csv'))
    return calcu_thick, measure_thick


def glc_inventory2oggm(gi_gdf, region='13', version='60', 
                           ids=None, id_suffix=None, save_special_columns=None,
                           assigned_col_values=None,):
    """
    Parameters
    ----------
    gi_gdf : :py:geopandas.GeoDataFrame
        geopandas opened glacier inventory shapefile
    region : str
        Glacier RGI region code . The default is '13'.
    version : str
        Glacier inventory version code. The default is '06'.
    ids : list of integer, each element should not beyond 5 digits
        Assign special id number to each glacier. The default is None, use the
        glacier order as the id number
    id_suffix : str or None
         Ad a suffix to the glacier id. The default is None, no suffix
    save_special_columns : list or str, columns name in the original gi_gdf
        Whether save special columns in the output data.
        The default is None.
    assigned_col_values : dict
        Add new columns or change the value of existed colunms. 
        The default is None.

    Returns
    -------
    gi_gdf : :py:geopandas.GeoDataFrame
        with same columns as RGI 

    """

    if ids:
        if len(ids) != len(gi_gdf):
            raise ValueError("The length of 'ids' do not match with 'gi_gdf'")
    else:
        ids = range(len(gi_gdf))
    
    if not id_suffix:
        id_suffix=''
    else:
        id_suffix = '_' + id_suffix
    id_ = ['RGI{}-{}.{:0>5d}{}'.format(version, region, i, id_suffix) 
           for i in ids]
    
    gi_gdf = gi_gdf.to_crs({'init': 'epsg:4326'})
    
    columns = ['RGIId', 'GLIMSId', 'BgnDate', 'EndDate', 'CenLon', 'CenLat',
       'O1Region', 'O2Region', 'Area', 'Zmin', 'Zmax', 'Zmed', 'Slope',
       'Aspect', 'Lmax', 'Status', 'Connect', 'Form', 'TermType',
       'Surging', 'Linkages', 'Name', 'check_geom', 'geometry']
    if save_special_columns:
        if type(save_special_columns) is str:
            columns.append(save_special_columns)
        elif type(save_special_columns) is list:
            [columns.append(column) for column in save_special_columns]
    gi_gdf = gi_gdf.reindex(columns=columns)
    
    if assigned_col_values:
        for key in assigned_col_values.keys():
            gi_gdf[key]=assigned_col_values[key]
    
    gi_gdf['RGIId'] = id_
    gi_gdf['CenLon'] = gi_gdf.geometry.centroid.x
    gi_gdf['CenLat'] = gi_gdf.geometry.centroid.y
    gi_gdf['GLIMSId'] = ['G{0:0>6d}E{1:0>5d}N'.format(round(x*1e3), 
                                                      round(y*1e3)) 
                            for x, y in zip(gi_gdf.CenLon, gi_gdf.CenLat)]
    gi_gdf['BgnDate'] = '20009999'
    gi_gdf['EndDate'] = '-9999999'
    gi_gdf['O1Region'] = '13'
    gi_gdf['O2Region'] = '1'
    gi_gdf[['Area', 
            'Zmin', 
            'Zmax', 
            'Zmed', 
            'Slope', 
            'Aspect', 
            'Lmax']] = np.full((len(gi_gdf), 7), -9999)
    gi_gdf[['Status', 
            'Connect', 
            'Form', 
            'TermType']] = np.full((len(gi_gdf), 4), 0)
    gi_gdf['Surging'] = 9
    gi_gdf['Linkages'] = 1
    gi_gdf['check_geom'] = None
    name_list = np.where(gi_gdf.Name.isnull(), '', gi_gdf.Name.values)
    gi_gdf['Name'] = name_list
    
    return gi_gdf


def get_rgi_region(level, region):
    """
    Get rgi region and sub_region

    Parameters
    ----------
    level : str
        '1' for RGI Region
        '2' for RGI Sub-Region
    region : str
        0~19: RGI Region code

    Returns
    -------
    rdf : geopandas.GeoDataFrame
        target region dataframe

    """
    
    rpath = os.path.join(utils.get_rgi_dir(version='6'), '00_rgi60_regions')
    f_name = f'00_rgi60_O{level}Regions.shp'
    rdf = gpd.read_file(os.path.join(rpath, f_name))
    if level == '1':
        rdf = rdf[rdf.RGI_CODE==int(region)].copy()
    elif level =='2':
        rdf = rdf[rdf.RGI_CODE.str.contains(region+'-')].copy()
    else:
        raise ValueError(f"No '{level}' level!")

    return rdf


def get_farinotti_file_path(rgiid, model_type, path=None):
    """
    locate farinotti et. al. (2019) ice thickness data

    Parameters
    ----------
    rgiid : str
    model_type : str:['0', '1', '2', '3', '4', '5']
        '0' for Composite result
        '1' for HF model results
        '2' for GlabTop2 model results
        '3' for OGGM model results
        '4' for Johannes et. al. (2017) model results
        '5' for RAAJ Ramsankaran et. al. (2018) model results
    path : str
        root directory for the data. The default is None.

    Returns
    -------
    outpath : str

    """

    region = rgiid.split('.')[0]
    if not path:
        path = '/home/lifei/Data/Farinotti_2019'
        if not os.path.exists(path):
            path = '/home/lifei/Data/Ice_Thickness_Farinotti'
    if model_type =='0':
        dir_name = 'composite_thickness_RGI60-all_regions'
        f_name = rgiid+'_thickness.tif'
    else:
        dir_name = f'results_model_{model_type}'
        f_name = 'thickness_'+rgiid+'.tif'
    outpath = os.path.join(path, dir_name, region, f_name)
    
    if not os.path.exists(outpath):
        outpath = None

    return outpath


def _extract_farinotti_thick(thick_df, path, thick_col=None, mask_df=None):
    """
    Extract farinotti simulated ice thickness according measured ice thickness

    Parameters
    ----------
    thick_df : geopandas.GeoDataFrame
        point thickness data
    path : str
        Farinotti data path
    mask_df : geopandas.GeoDataFrame
        Clip thick_df in the target region

    Returns
    -------
    mthick : np.ndarray
        measured thickness (if multiple measure point in the same grid, they 
        will be averaged)
    cthick : np.ndarray
        simulated thickness

    """
    
    ds = salem.open_xr_dataset(path)
    grid = ds.salem.grid
    mt_arr = _gridded_measured_thickness_point(grid, thick_df, 
                                               thick_col=thick_col, 
                                               mask_df=mask_df)
    ct_arr = ds.data.values
    
    mthick = mt_arr[np.where(np.isfinite(mt_arr))]
    cthick = ct_arr[np.where(np.isfinite(mt_arr))]
    
    return mthick, cthick


def extract_farinotti_thick(thick_df, gdir=None, name=None, rgi_id=None, thick_col=None, mask_df=None):
    """

    Parameters
    ----------
    gdir
    thick_df
    thick_col
    mask_df

    Returns
    -------

    """

    if gdir:
        rgi_id = gdir.rgi_id
    else:
        if not rgi_id:
           raise ValueError("No 'rgi_id' or 'gdir' was provided!")
    paths = [get_farinotti_file_path(rgiid=rgi_id, model_type=str(i)) for i in range(5)]
    if not name:
        if gdir:
            name = gdir.name
        else:
            name = rgi_id
    cthicks = []
    for path in paths:
        mthick, cthick = _extract_farinotti_thick(thick_df=thick_df, thick_col=thick_col, mask_df=mask_df, path=path)
        cthicks.append(cthick)
    df = pd.DataFrame(dict(mea_thick=mthick,
                           cal_thick_0=cthicks[0],
                           cal_thick_1=cthicks[1],
                           cal_thick_2=cthicks[2],
                           cal_thick_3=cthicks[3],
                           cal_thick_4=cthicks[4]))
    return df


def tmerc_proj(lon):
    proj_str = f'+proj=tmerc +lat_0=0 +lon_0={lon} +k=0.9996 ' + \
        '+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs'
    
    return proj_str

def proj_gdf2tmerc(gdf):
    
    gdf = gdf.copy()
    cenlon = np.mean([gdf.geometry.bounds.maxx.max(), 
                      gdf.geometry.bounds.minx.min()])
    gdf = gdf.to_crs(tmerc_proj(cenlon))
    
    return gdf
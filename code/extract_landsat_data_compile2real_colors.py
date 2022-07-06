#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:00:28 2020

@author: keeptg
"""

import tarfile, os
import rasterio

def extract_tar_gz(input_path, output_path):
    """
    Extract the '*.tar.gz' compressed file
    
    Parameters
    ----------
    input_path : str: the path of compressed file
    output_path : str: the path of extracted file
    
    Returns
    -------
    None.

    """
    tar = tarfile.open(input_path, "r:gz")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_names = tar.getnames()
    for file_name in file_names:
        tar.extract(file_name, output_path)
    tar.close()


root_path = '/home/keeptg/Data/Study_in_Innsbruck/Tienshan'
landsat_path = os.path.join(root_path, 'Landsat8')
files = os.listdir(landsat_path)
out_path = os.path.join(root_path, 'Landsat8_extracted')
for file in files:
    path = os.path.join(landsat_path, file)
    fs = os.listdir(path)
    if fs:
        fs = os.listdir(path)
        for f in fs:
            input_path = os.path.join(path, f)
            output_path = os.path.join(out_path, file, f.split('.')[0])
            try:
                extract_tar_gz(input_path, output_path)
            except:
                print(f'Meet Error on {input_path}')
                continue
            

l8_bands = ['B4.TIF', 'B3.TIF', 'B2.TIF']
l7_bands = ['B3.TIF', 'B2.TIF', 'B1.TIF']
path = os.path.join(root_path, 'Landsat8_extracted')
r_outpath = os.path.join(root_path, 'Landsat_Real_color')
for dir1 in os.listdir(path):
    path1 = os.path.join(path, dir1)
    for dir2 in os.listdir(path1):
        path2 = os.path.join(path1, dir2)
        if dir2.split('_')[0] == 'LC08':
            bands = l8_bands
            # dtype = rasterio.uint16
        elif dir2.split('_')[0] == 'LE07':
            bands = l7_bands
            # dtype = rasterio.uint8
        else:
            raise ValueError(f"Can't find the file type of {dir2}")

        rband = dir2 + '_' + bands[0]
        gband = dir2 + '_' + bands[1]
        bband = dir2 + '_' + bands[2]
        
        r = rasterio.open(os.path.join(path2, rband))
        b = rasterio.open(os.path.join(path2, bband))
        g = rasterio.open(os.path.join(path2, gband))
        
        profile = r.profile
        profile.update(count=3)
        file_name = dir2.split('_')[0] + '_' + dir2.split('_')[3] + '.tif'
        outpath = os.path.join(r_outpath, dir1, dir2)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        with rasterio.open(os.path.join(outpath, file_name),
                                        'w', **profile) as dst:
            dst.write_band(1, r.read(1))
            dst.write_band(2, g.read(1))
            dst.write_band(3, b.read(1))
        
















import os
import sys
import numpy as np
import pandas as pd

import geopandas as gpd

from oggm import cfg, tasks, workflow, utils
from path_config import *

geodf = utils.get_geodetic_mb_dataframe()

rgidf = gpd.read_file(os.path.join(data_dir, 'rgi_glathida'))
rgidf.drop(index=6, inplace=True, axis=0)

geodf = geodf.loc[rgidf.RGIId]
geodf2010 = geodf[geodf.period=='2000-01-01_2010-01-01']
geodf2020 = geodf[geodf.period=='2000-01-01_2020-01-01']
geodf2010.mean()
geodf2020.mean()
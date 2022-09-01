#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python


# # CONUS404_pr-changes.ipynb

# In[2]:


'''File name: CONUS404_pr-changes.ipynb
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 19.05.2022
    Date last modified: 19.05.2022

    ############################################################## 
    Purpos:

    - Rean in hourly precipitation data from CONUS404 
    - Save the data at lower precission to make it easier accessible
    - Calculate changes in the hourly precipitation distribution 

'''


# In[3]:


import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import glob
import os
from pdb import set_trace as stop
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import median_filter
from scipy.ndimage import label
from matplotlib import cm
from scipy import ndimage
import random
import scipy
import pickle
import datetime
import pandas as pd
import subprocess
from calendar import monthrange
import pandas as pd
import datetime
import sys 
import shapefile as shp
import matplotlib.path as mplPath
from scipy.stats import norm
import matplotlib.gridspec as gridspec
# from mpl_toolkits.basemap import Basemap, cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pylab import *
import string
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import shapefile
from calendar import monthrange
from tqdm import tqdm
import xarray as xr

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
from matplotlib.colors import LogNorm
from cartopy.feature import NaturalEarthFeature
import cartopy.feature as cf

from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
                 cartopy_ylim, latlon_coords)

def read_shapefile(sf):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' 
    column holding the geometry information. This uses the pyshp
    package
    """
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df

#### speed up interpolation
import scipy.interpolate as spint
import scipy.spatial.qhull as qhull
import numpy as np

def interp_weights(xy, uv,d=2):
    tri = qhull.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(values, vtx, wts):
    return np.einsum('nj,nj->n', np.take(values, vtx), wts)

def deiscretice_timeseries(DATA,
                          bucked_size):
#     Discrete_timeseries = np.copy(DATA); Discrete_timeseries[:] = np.nan
#     for tt in range(len(DATA)):
#         if ~np.isnan(DATA[tt]) == True:   
#             INT, REST = divmod(DATA[tt], bucked_size)
#             Discrete_timeseries[tt] = INT * bucked_size
#             if tt != len(DATA)-1:
#                 DATA[tt+1] = DATA[tt+1]+REST
#     return Discrete_timeseries

    if len(DATA.shape) == 1:
        # make data 2D
        DATA = DATA[:,None]
    Discrete_timeseries = np.copy(DATA); Discrete_timeseries[:] = np.nan
    for tt in tqdm(range(DATA.shape[0])):
        INT, REST = np.apply_along_axis(np.divmod, 0, DATA[tt,:], bucked_size)
        FIN = ~np.isnan(INT)
        Discrete_timeseries[tt,:] = INT * bucked_size
        if tt != len(DATA)-1:
            DATA[tt+1,FIN] = DATA[tt+1,FIN]+REST[FIN]
    return Discrete_timeseries


# In[4]:


# ================================
# BUKOFSKY REGION
# Add the subregions
import geopandas as gpd
from tqdm import tqdm

REGIONS = [ 'Appalachia.shp',
            'CPlains.shp',
            'DeepSouth.shp',
            'GreatBasin.shp',
            'GreatLakes.shp',
            'Mezquital.shp',
            'MidAtlantic.shp',
            'NorthAtlantic.shp',
            'NPlains.shp',
            'NRockies.shp',
            'PacificNW.shp',
            'PacificSW.shp',
            'Prairie.shp',
            'Southeast.shp',
            'Southwest.shp',
            'SPlains.shp',
            'SRockies.shp']

REGIONS_names = [ 'Appalachia',
            'Central Plains',
            'Deep South',
            'Great Basin',
            'Great Lakes',
            'Mezquital',
            'Mid-Atlantic',
            'North-Atlantic',
            'Northern Plains',
            'Northern Rockies',
            'Pacific Northwest',
            'Pacific Southwest',
            'Prairie',
            'Southeast',
            'Southwest',
            'Southern Plains',
            'Southern Rockies']


# In[5]:


##############################################################
#                READ CONUS404 CONSTANT FIELDS
sLon='XLONG'
sLat='XLAT'
sOro='HGT'
sLSM='LANDMASK'
sPlotDir = ''
GEO_EM_D1 = '/glade/campaign/ncar/USGS_Water/CONUS404/wrfconstants_d01_1979-10-01_00:00:00.nc4'

ncid=Dataset(GEO_EM_D1, mode='r') # open the netcdf
Lon=np.squeeze(ncid.variables[sLon][:])
Lat=np.squeeze(ncid.variables[sLat][:])
Height4=np.squeeze(ncid.variables[sOro][:])
LSM=np.squeeze(ncid.variables[sLSM][:])
ncid.close()


# In[6]:


DataFolder = '/glade/campaign/mmm/c3we/prein/CONUS404/data/MonthlyData/'
SaveFolder = '/glade/campaign/mmm/c3we/prein/CONUS404/data/CONUS404_processed_data/'

StartDay = datetime.datetime(1980, 1, 1, 0)
StopDay = datetime.datetime(2019, 12, 31, 23)
TimeHH = pd.date_range(StartDay, end=StopDay, freq='1h')
TimeMM = pd.date_range(StartDay, end=StopDay, freq='M')
Years = np.unique(TimeMM.year)

Seasons = ['annual','DJF','MAM','JJA','SON']
rgiSeasons = [range(1,13,1),
                 [1,2,12],
                 [3,4,5],
                 [6,7,8],
                 [9,10,11]]

dry_threshold = 0.05 # mm/h


# In[7]:


MaskFile = 'Burkofski_Regions_CONUS404.npz'
if os.path.exists(MaskFile) == False:
    rgiStatsInBasins = []
    MaskStations = np.zeros((Lon.shape[0],Lon.shape[1])); MaskStations[:] = np.nan
    rgrGridCells=[(Lon.flatten()[ii],Lat.flatten()[ii]) for ii in range(len(Lon.flatten()))]
    for re in tqdm(range(len(REGIONS))):
        data = gpd.read_file('/glade/u/home/prein/papers/2021_Hist-Ext-PR-Changes/shapefiles/Bukovski-Regions/'+REGIONS[re])
        Coordinates = data['geometry']
        for sf in range(len(data)):
            TEST = np.array(Coordinates[sf].exterior.coords.xy)
            ctr=TEST.T
            grPRregion=mplPath.Path(ctr)
            TMP=np.array(grPRregion.contains_points(rgrGridCells))
        TMP = np.reshape(TMP, (Lon.shape[0], Lon.shape[1]))
        MaskStations[TMP==1] = re+1
    #     MaskStations = np.append(MaskStations,[re+1]*len(iStationSelect))
    MaskStations = MaskStations.astype('int')
    
    np.savez(MaskFile,
            MaskStations = MaskStations,
            Lon = Lon,
            Lat = Lat)
else:
    DATA = np.load(MaskFile)
    MaskStations = DATA['MaskStations']
    Lon = DATA['Lon']
    Lat = DATA['Lat']
MaskStations[MaskStations < 0] = 0

Region_indices = ndimage.find_objects(MaskStations)
# Region_indices = ndimage.find_objects(MaskStations)


# ### READ HOURLY CONUS404 PRECIPITATION DATA

# In[8]:


Save_file = SaveFolder+'PREC_ACC_NC_'+str(TimeMM[0].year)+str(TimeMM[0].month).zfill(2)+'-'+str(TimeMM[-1].year)+str(TimeMM[-1].month).zfill(2)+'.npz'

if os.path.exists(Save_file) == False:
    CONUS404_hourly_pr = np.zeros((len(TimeHH), Lon.shape[0], Lon.shape[1]), dtype=np.float16)
    for mm in tqdm(range(len(TimeMM))):
        YYYY = TimeMM[mm].year
        MM = TimeMM[mm].month
        rgiHours = (TimeHH.year == YYYY) & (TimeHH.month == MM)
        File_act = DataFolder + 'PREC_ACC_NC_'+str(YYYY)+str(MM).zfill(2)+'_CONUS404.nc'
        ncid=Dataset(File_act, mode='r') # open the netcdf
        CONUS404_hourly_pr[rgiHours,:,:] = np.array(np.squeeze(ncid.variables['PREC_ACC_NC'][:]), dtype=np.float16)
        ncid.close()
        np.savez(Save_file,
            CONUS404_hourly_pr = CONUS404_hourly_pr,
            TimeHH = TimeHH,
            Lon = Lon,
            Lat = Lat)
else:
    print('Load data from '+Save_file)
    DATA = np.load(Save_file)
    CONUS404_hourly_pr = DATA['CONUS404_hourly_pr']
    TimeHH = pd.to_datetime(DATA['TimeHH'])
    Lon = DATA['Lon']
    Lat = DATA['Lat']



NN = 100
bins = np.array([0]+ [np.exp(np.log(0.005) + (ii* ((np.log(120)-np.log(0.005))**2/(NN)) )**0.5 ) for ii in range(NN)])
bins_cent = (bins[1:]+bins[:-1])/2
# bins = np.exp(np.linspace(0,4.5,101))-1


# In[237]:


# Calculate fractional contribution of precipitation
bootstratp_nr = 100
pr_frac_contributions = np.zeros((NN, 2, bootstratp_nr, len(REGIONS))); pr_frac_contributions[:] = np.nan
import random
half_rec_len = int(CONUS404_hourly_pr.shape[0]/2)
SampleSize = 2000

for re in range(len(REGIONS)):
    print('work on '+REGIONS[re])
    for bs in tqdm(range(bootstratp_nr)):
        for pe in range(2):
            if pe == 0:
                sample = random.sample(range(0, half_rec_len), SampleSize)
            elif pe == 1:
                sample = random.sample(range(half_rec_len, CONUS404_hourly_pr.shape[0]-1), SampleSize)
            
            DATA_test = CONUS404_hourly_pr[sample,Region_indices[re][0].start:Region_indices[re][0].stop, 
                  Region_indices[re][1].start:Region_indices[re][1].stop] 
            DATA_test = np.array(DATA_test[:, MaskStations[Region_indices[re]] == re+1].flatten(), dtype=np.float32)
            pr_frac_contributions[:,pe,bs,re] = np.array([np.sum(DATA_test[(DATA_test >= bins[ii]) & (DATA_test <= bins[ii+1])]) for ii in range(len(bins)-1)])


np.savez(SaveFolder+'Fractional_Contribution_changes.npz',
        pr_frac_contributions = pr_frac_contributions,
        REGIONS_names = REGIONS_names,
        bins = bins,
        bootstratp_nr = bootstratp_nr,
        SampleSize = SampleSize)
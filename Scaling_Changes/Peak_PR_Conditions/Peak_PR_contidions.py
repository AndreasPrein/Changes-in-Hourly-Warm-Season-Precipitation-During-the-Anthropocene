#!/usr/bin/env python

# # Scaling_Changes.ipynb

# In[1]:


'''File name: Peak_PR_contidions.py
    Author: Andreas Prein
    E-mail: prein@ucar.edu
    Date created: 18.06.2022
    Date last modified: 18.06.2022

    ############################################################## 
    Purpos:

    - Rean in hourly precipitation data from CONUS404 
    - Read in hourly target variable from CONUS404
    - Calculate top 10 precipitation hours per year and the corresponding target variables

'''


# In[2]:


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


# In[3]:


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


# In[4]:


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


DataFolder = '/glade/campaign/mmm/c3we/prein/CONUS404/data/MonthlyData/'
SaveFolder = '/glade/campaign/mmm/c3we/prein/CONUS404/data/CONUS404_processed_data/'

StartDay = datetime.datetime(1980, 1, 1, 0)
StopDay = datetime.datetime(2019, 12, 31, 23)
TimeHH = pd.date_range(StartDay, end=StopDay, freq='1h')
TimeMM = pd.date_range(StartDay, end=StopDay, freq='M')
Years = np.unique(TimeMM.year)
YYYY = int(sys.argv[1])
VAR = str(sys.argv[2]) # options are 'TD2'

print('---------------------------')
print(VAR+' '+str(YYYY))

# Seasons = ['annual','DJF','MAM','JJA','SON']
# rgiSeasons = [range(1,13,1),
#                  [1,2,12],
#                  [3,4,5],
#                  [6,7,8],
#                  [9,10,11]]

se = 0
Seasons = ['MJJASO','May-Jun','Jul-Aug','Sep-Oct']
rgiSeasons = [[5,6,7,8,9,10],
                 [5,6],
                 [7,8],
                 [9,10]]

dry_threshold = 0.1 # mm/h
lag_hours = 2 # hours before the pr event that dT is taken
top_events = 10 # get 10 most extreme hourly pr events per grid cell
top_events_distance = 24 # at least 24-hours have to be between each pr event

rgiHours = (TimeHH.year == YYYY) & np.isin(TimeHH.month, rgiSeasons[se])
timeHH_yy = TimeHH[rgiHours]
prec_yy = np.array(np.zeros((len(timeHH_yy), Lon.shape[0], Lon.shape[1])), dtype=np.float32); prec_yy[:] = np.nan
VAR_yy = np.copy(prec_yy)

# for mm in tqdm(range(12)):
for mm in tqdm(range(len(rgiSeasons[se]))):
    MM = rgiSeasons[se][mm]
    time_mm_in_yy = timeHH_yy.month == MM

    # read precipitation
    File_act = DataFolder + 'PREC_ACC_NC_'+str(YYYY)+str(MM).zfill(2)+'_CONUS404.nc'
    ncid=Dataset(File_act, mode='r') # open the netcdf
    prec_yy[time_mm_in_yy,:,:] = np.array(np.squeeze(ncid.variables['PREC_ACC_NC'][:]), dtype=np.float32)
    ncid.close()

    # read secondary variable that might be connected to PREC
    File_act = DataFolder + VAR + '_'+str(YYYY)+str(MM).zfill(2)+'_CONUS404.nc'
    ncid=Dataset(File_act, mode='r') # open the netcdf
    VAR_yy[time_mm_in_yy,:,:] = np.array(np.squeeze(ncid.variables[VAR][:]), dtype=np.float32)
    ncid.close()


# ### Get the top N events that are at least the "top_events_distance" appart

# In[2]:


Sorted = np.argsort(prec_yy[:,:,:], axis = 0)[::-1]
extreme_ind = np.zeros((100,Sorted.shape[1],Sorted.shape[2])); extreme_ind[:] = np.nan
extreme_ind[0,:,:] = Sorted[0,:,:]
for ex in tqdm(range(1, extreme_ind.shape[0], 1)):
    temp_ext_ind = Sorted[ex,:,:]
    time_dist = np.abs(temp_ext_ind[None,:,:] - extreme_ind[:ex,:,:])
    too_close = np.nanmin(time_dist, axis=0) < top_events_distance
    temp_ext_ind[too_close] = -9999
    extreme_ind[ex,:,:] = temp_ext_ind
extreme_ind[extreme_ind < 0] = np.nan

extr_pr_top_ind = np.zeros((10,Sorted.shape[1],Sorted.shape[2])); extr_pr_top_ind[:] = np.nan
for la in tqdm(range(Sorted.shape[1])):
    for lo in range(Sorted.shape[2]):
        ind_tmp = extreme_ind[:,la,lo][~np.isnan(extreme_ind[:,la,lo])][:10]
        extr_pr_top_ind[:len(ind_tmp),la,lo] = ind_tmp


# In[3]:


extr_pr_top_ind = extr_pr_top_ind.astype('int')


# In[19]:


pr_extr = np.zeros((extr_pr_top_ind.shape)); pr_extr[:] = np.nan
for ii in range(extr_pr_top_ind.shape[0]):
    # https://stackoverflow.com/questions/45335535/use-2d-matrix-as-indexes-for-a-3d-matrix-in-numpy
    aa = extr_pr_top_ind[ii,:,:]
    aa[aa < 0] = 0
    m,n = aa.shape
    I,J = np.ogrid[:m,:n]
    pr_extr[ii,:,:] = prec_yy[aa, I, J]
    pr_extr[ii,extr_pr_top_ind[ii,:,:] < 0] = np.nan


# In[26]:


VAR_extr = np.zeros((extr_pr_top_ind.shape)); VAR_extr[:] = np.nan
for ii in range(extr_pr_top_ind.shape[0]):
    # https://stackoverflow.com/questions/45335535/use-2d-matrix-as-indexes-for-a-3d-matrix-in-numpy
    aa = extr_pr_top_ind[ii,:,:]
    aa == aa - lag_hours
    aa[aa < 0] = 0
    m,n = aa.shape
    I,J = np.ogrid[:m,:n]
    VAR_extr[ii,:,:] = VAR_yy[aa, I, J]
    VAR_extr[ii,extr_pr_top_ind[ii,:,:] < 0] = np.nan



np.savez(SaveFolder+'pr_vs_VARS/'+str(YYYY)+'_'+Seasons[se]+'_pr-vs-'+VAR+'_gridcells.npz',
        dry_threshold = dry_threshold,
        lag_hours = lag_hours,
        pr_extr = pr_extr,
         VAR_extr = VAR_extr,
         VAR = VAR,
         top_events_distance = top_events_distance,
         extr_pr_top_ind = extr_pr_top_ind,
         TimeHH = TimeHH)








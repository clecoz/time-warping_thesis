import netCDF4 as nc
import numpy as np
#import scipy.signal as sc
import scipy.optimize as op
from scipy import interpolate
import os
import pandas as pd
import logging
import matplotlib.pyplot as plt
import pylab
import glob
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.dates as dt
import matplotlib.animation as animation

from morphing_function import *
from registration_approach2 import *
from registration_approach1 import *
from registration_approach3 import *

import time
import pyproj
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from scipy.sparse import csr_matrix

#=======================================================================================================================
# Synthetic case
#=======================================================================================================================

# Choose case
case = 'case_spacetime'

# Choose experience
exp = "full"    #"full" or "interpolated"

# Choose where to save results
folder_result = "{}/results/".format(case).replace('.','p')
if not os.path.exists(folder_result):
    os.makedirs(folder_result)

# Choose if you want to plot or compute statistics
plot = True
stats = True
threshold = 10   # in mm/h, used to compute the position and timing error


#==================================================================
# Choose registration parameter

# Choose registration approach
reg = "a2"          # Possible choice: "a1", "a2" and "a3"

# Choose regulation coefficients
c1 = 0.1
c2 = 1
c3 = 1

# Choose number of morphing grids
I = 3


#==================================================================
# Prepare data from the chosen case
print('Prepare data')

# Files containing the data
TAHMO_file = './{}/v.csv'.format(case)
TAHMO_sta_file = './{}/v_stations.csv'.format(case)
IMERG_file = './{}/u.csv'.format(case)
IMERG_sta_file = './{}/u_stations.csv'.format(case)
file_coord = './{}/coord_stations.csv'.format(case)

# Dimensions
nt = 25
nx = 65
ny = 65

# Data to be corrected (IMERG-Late)
u_domain = np.loadtxt(IMERG_file, delimiter=',').reshape((25,65,65))[:,14:-14,14:-14]    # We select a subdomain of 37 by 37 grid points. The goal is to make this case similar to the southern Ghana case (in term of size)
lon_u = np.array(np.arange(-4.75, 1.75, 0.1))
lat_u = np.array(np.arange(3.25, 9.75, 0.1))
npts = lon_u.shape[0]
u_station = np.loadtxt(IMERG_sta_file, delimiter=',')

# Reference data (TAHMO)
df_coord = pd.read_csv(file_coord, index_col=0)
lon_v = df_coord.loc['lon'].values
lat_v = df_coord.loc['lat'].values
v_domain = np.loadtxt(TAHMO_file, delimiter=',').reshape((-1,65,65))
v_station = np.loadtxt(TAHMO_sta_file, delimiter=',')

# Coordinates
t = np.array(range(0, nt))
x = lon_u
y = lat_u
nx = len(x)
ny = len(y)


#==================================================================
# Pre-processing
print('\nStart pre-processing')

# Resampling time (artificially increase the number of points)
t_resample = np.linspace(0,nt-1,241)
nt = len(t_resample)
u = np.zeros(shape=(nt,npts))
v = np.zeros(shape=(nt,npts))
for k in range(npts):
    u[:,k] = np.interp(t_resample,t,u_station[:,k])
    v[:,k] = np.interp(t_resample,t,v_station[:,k])


#================================================================
# Define the correlation between stations

# Convert coordinates
proj_latlon = pyproj.Proj(init="epsg:4326")
proj_utm = pyproj.Proj(init="epsg:32630")
x,y = pyproj.transform(proj_latlon,proj_utm,lon_v, lat_v)
X = (1/1000)*np.array([x,y]).T

# Compute distances
d = pdist(X, metric='euclidean')

# Exponential variogram
psill = 1
range_ = 150
nugget = 0
vario = psill * (1. - np.exp(-d/(range_/3.))) + nugget
space_corr = 1 - vario

# Combinatorial matrix
Acomb = np.zeros((len(list(combinations(range(npts), 2))),npts))
nit = 0
for k, j in combinations(range(npts), 2):
    Acomb[nit,k] = -1
    Acomb[nit, j] = 1
    nit += 1
Acomb = csr_matrix(Acomb)


#================================================================
# Registration
print("\nStart regisration")

ks = 'all'          # this parameter is used for the cross-validation in the case of the southern Ghana case

start = time.time()

if reg == "a1":
    Tt = registration_a1(u,v,t_resample,I,c1,c2,folder_result,ks)
elif reg == "a2":
    Tt = registration_a2(u,v,t_resample,I,c1,c2,folder_result,ks)
elif reg == "a3":
    Tt = registration_a3(u,v,t_resample,I,c1,c2,c3,Acomb,space_corr,folder_result,ks)
else:
    print("Error (wrong ""reg""): this approach does not exist.")


end = time.time()
print('\n Elapsed time for ks {}: {}'.format(ks,end - start))


#================================================================
# Warping
print("\nWarping")

u_warped_station = mapped(u_station, t, Tt, I)



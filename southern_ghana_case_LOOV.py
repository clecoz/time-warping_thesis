from registration_approach2 import *
from registration_approach1 import *
from registration_approach3 import *

import time
import pyproj
import time
import pyproj
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
from scipy.sparse import csr_matrix
import os
import pandas as pd
from pykrige import OrdinaryKriging
import math
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.colors as cls
from matplotlib.colors import ListedColormap

#=======================================================================================================================
# Synthetic case
#=======================================================================================================================

# Choose case
case = '20180422'
folder = case

# Choose where to save results
folder_result = "{}/results/".format(case).replace('.','p')
if not os.path.exists(folder_result):
    os.makedirs(folder_result)

# Choose if you want to plot or compute statistics
plot = True
stats = True
threshold = 0.1   # in mm/h, used to compute the position and timing error


#==================================================================
# Choose registration parameter

# Choose registration approach
reg = "a1"          # Possible choice: "a1", "a2" and "a3"

# Choose regulation coefficients
c1 = 0.1
c2 = 1
cs = 1

# Choose number of morphing grids
I = 3


#==================================================================
# Prepare data from the chosen case
print('Prepare data')

# Files containing the data
TAHMO_file = './{}/TAHMO_{}.csv'.format(folder, case)
IMERG_file = './{}/IMERG_all_{}.csv'.format(folder, case)
IMERG_station_file = './{}/IMERG_{}.csv'.format(folder, case)
file_coord_sta = './{}/coord_stations.csv'.format(folder, case)
file_coord_IMERG =  './{}/coord_IMERG.csv'.format(folder)

# Read data from files
df_TAHMO = pd.read_csv(TAHMO_file, index_col=0)
df_IMERG = pd.read_csv(IMERG_file, index_col=0)
df_IMERG_sta = pd.read_csv(IMERG_station_file, index_col=0)
df_coord_sta = pd.read_csv(file_coord_sta, index_col=0)
df_coord_IMERG = pd.read_csv(file_coord_IMERG, index_col=0)

# Data to be corrected (IMERG-Late)
u_domain = df_IMERG.values.reshape((-1,37,37))
lon_u = df_coord_IMERG.loc['lon'].values
lat_u = df_coord_IMERG.loc['lat'].values
u_station = df_IMERG_sta.values

# Reference data (TAHMO)
lon_v = df_coord_sta.loc['lon'].values
lat_v = df_coord_sta.loc['lat'].values
v_station = df_TAHMO.values
station_ID = df_TAHMO.columns.values

# Coordinates
nt, ns = v_station.shape
t = np.array(range(0, nt))
datetime = df_TAHMO.index.values
x = lon_u
y = lat_u
nx = len(x)
ny = len(y)
npts = nx*ny


#==================================================================
# Pre-processing
print('\nStart pre-processing')

# Resampling time (artificially increase the number of points)
t_resample = np.linspace(0,nt-1,241)
ntr = len(t_resample)
u_all = np.zeros(shape=(ntr,ns))
v_all = np.zeros(shape=(ntr,ns))
for k in range(ns):
    u_all[:,k] = np.interp(t_resample,t,u_station[:,k])
    v_all[:,k] = np.interp(t_resample,t,v_station[:,k])


#================================================================
# Start LOOV
# We are looping on the stations, removing one from the input.
# This station can then be used later for validation.


# Create array where we will save the maapings for each iteration
if reg == "a1" or reg == "a3":
    Tt_loov = np.zeros((2 ** I + 1, ns-1, ns))
elif reg == "a2":
    Tt_loov = np.zeros((2 ** I + 1, ns))
else:
    print("Error (wrong ""reg""): this approach does not exist.")


# Start iteration
for ks in range(ns):
    print('\n Remove station {} ({})'.format(ks,station_ID[ks]))

    # Remove one station from input
    u_loov = np.delete(u_all, ks, axis=1)
    v_loov = np.delete(v_all, ks, axis=1)
    lons_loov = np.delete(lon_v, ks)
    lats_loov = np.delete(lat_v, ks)


    # ---------------------------------------------
    # Define the correlation between stations

    # Convert coordinates
    proj_latlon = pyproj.Proj(init="epsg:4326")
    proj_utm = pyproj.Proj(init="epsg:32630")
    x_proj,y_proj = pyproj.transform(proj_latlon, proj_utm, lons_loov, lats_loov)
    X = (1/1000)*np.array([x_proj, y_proj]).T

    # Compute distances
    d = pdist(X, metric='euclidean')

    # Exponential variogram
    psill = 1
    range_ = 150
    nugget = 0
    vario = psill * (1. - np.exp(-d/(range_/3.))) + nugget
    space_corr = 1 - vario

    # Combinatorial matrix
    Acomb = np.zeros((len(list(combinations(range(ns-1), 2))),ns-1))
    nit = 0
    for k, j in combinations(range(ns-1), 2):
        Acomb[nit,k] = -1
        Acomb[nit, j] = 1
        nit += 1
    Acomb = csr_matrix(Acomb)


    # ---------------------------------------------
    # Registration
    if reg == "a1":
        Tt = registration_a1(u_loov, v_loov, t_resample, I, c1, c2, folder_result, ks)
        Tt_loov[:, :, ks] = Tt
    elif reg == "a2":
        Tt = registration_a2(u_loov, v_loov, t_resample, I, c1, c2, folder_result, ks)
        Tt_loov[:, ks] = Tt
    elif reg == "a3":
        Tt = registration_a3(u_loov, v_loov, t_resample, I, c1, c2, cs, Acomb, space_corr, folder_result, ks)
        Tt_loov[:, :, ks] = Tt
    else:
        print("Error (wrong ""reg""): this approach does not exist.")




#================================================================
# Post-processing and Warping
print("\nPost-processing and warping")


# Interpolate the mappings
Tt_station = np.zeros((2**I+1, ns))
Tt_grid = np.zeros((2**I+1, ns, npts))
for ks in range(ns):
    lons_loov = np.delete(lon_v, ks)
    lats_loov = np.delete(lat_v, ks)
    x_proj, y_proj = pyproj.transform(proj_latlon, proj_utm, lons_loov, lats_loov)
    X = (1 / 1000) * np.array([x_proj, y_proj])
    xt, yt = pyproj.transform(proj_latlon, proj_utm, lon_u, lat_u)
    xt *= (1 / 1000)
    yt *= (1 / 1000)
    xts, yts = pyproj.transform(proj_latlon, proj_utm, lon_v[ks], lat_v[ks])
    xts *= (1 / 1000)
    yts *= (1 / 1000)

    if reg == "a2":
        Tt_station[:, ks] = Tt.T.reshape(-1)
        for kpt in range(npts):
            Tt_grid[:, ks, kpt] = Tt.T.reshape(-1)

    elif reg == "a1" or reg == "a3":
        # Interpolate at the missing station
        for ki in range(2 ** I + 1):
            OK = OrdinaryKriging(X[0, :], X[1, :], Tt_loov[ki, :, ks], variogram_model='exponential',
                                 variogram_parameters={'sill': 1.0, 'range': 150, 'nugget': 0.0}, nlags=50, verbose=False,
                                 enable_plotting=False, weight=True, coordinates_type='euclidean')
            z, ss = OK.execute('grid', xts, yts)
            Tt_station[ki, ks] = z
        # Interpolate on the grid
        for ki in range(2**I+1):
            OK = OrdinaryKriging(X[0, :], X[1, :], Tt_loov[ki, :, ks], variogram_model='exponential',
                                 variogram_parameters={'sill': 1.0, 'range': 150, 'nugget': 0.0}, nlags=50,
                                 verbose=False,
                                 enable_plotting=False, weight=True, coordinates_type='euclidean')
            z, ss = OK.execute('grid', xt, yt)
            Tt_grid[ki, ks, :] = z.T.reshape(-1)


# Warped field at the station locations for the ns iteration
u_warped_station = mapped(u_all, t_resample, Tt_station, I)

# Warped field on the grids
#u_warped_station = mapped(u, t_resample, Tt_station, I)


#================================================================
# Statistics

if stats:
    print('\nStatistics')
    # The statistics are computed at the station locations where we have gauge measurements

    # Statistics before warping
    RMSE_before = np.sqrt(np.mean((u_all - v_all) ** 2))
    MAE_before = np.mean(np.abs(u_all - v_all))
    C_before = np.corrcoef(u_all.reshape(-1), v_all.reshape(-1))[0, 1]

    # Statistics after warping
    RMSE_after = np.sqrt(np.mean((u_warped_station - v_all) ** 2))
    MAE_after = np.mean(np.abs(u_warped_station - v_all))
    C_after = np.corrcoef(u_warped_station.reshape(-1), v_all.reshape(-1))[0, 1]

    print('#---------------------------------------#')
    print('#        |  RMSE  |  MAE  | Correlation #')
    print('#---------------------------------------#')
    print('# Before |  {:.2f}  |  {:.2f} |    {:.2f}     #'.format(RMSE_before, MAE_before,C_before))
    print('# After  |  {:.2f}  |  {:.2f} |    {:.2f}     #'.format(RMSE_after, MAE_after,C_after))
    print('#---------------------------------------#')


    # ---------------------------------------------
    # Timing error
    # Find indexes of the pixel with the max. rain
    tv = np.argmax(v_all, axis=0)
    tu = np.argmax(u_all, axis=0)
    tuw = np.argmax(u_warped_station, axis=0)

    # Compute avrage timing difference
    mask = (~(v_all < threshold).all(axis=0) * ~(u_all < 0.1).all(axis=0))
    timing_before = np.mean(np.abs(tu[mask] - tv[mask])) / 10
    timing_after = np.mean(np.abs(tuw[mask] - tv[mask])) / 10

    # Print results
    print("\nAverage timing error for threshold={}mm/h (sample number = {})".format(threshold, np.sum(mask)))
    print("Before: {:.2f} h".format(timing_before))
    print("After:  {:.2f} h".format(timing_after))


#================================================================
# Plotting

if plot:
    print("\nPlotting")

    # Coordinates
    xx, yy = np.meshgrid(lon_u, lat_u, indexing='ij')
    tc = range(0, nt, int((nt - 1) / 2 ** I))


    # Plot mapping (average)
    lon_m = np.asarray(list(lon_u) + [lon_u[-1] + 0.1, ]) - 0.05
    lat_m = np.array(list(lat_u) + [lat_u[-1] + 0.1, ]) - 0.05
    lon_o, lat_o = np.meshgrid(lon_m, lat_m, indexing='ij')
    fig, axarr = plt.subplots(1, 2 ** I + 1, figsize=(25, 5))
    for ki in range(0, 2 ** I + 1):
        Tt_grid_mean = np.mean(Tt_grid[ki,:,:], axis=0)

        m = Basemap(llcrnrlon=-3.5, llcrnrlat=4.5, urcrnrlon=0.5, urcrnrlat=8.5, resolution='l', projection='merc',
                    lat_1=20., lat_2=0, lat_0=10., lon_0=5., ax=axarr[ki])
        if ki == 0:
            m.drawparallels(np.arange(-80., 81., 1.), labels=[1, 0, 0, 0], fontsize=10)
        else:
            m.drawparallels(np.arange(-80., 81., 1.), labels=[0, 0, 0, 0], fontsize=10)
        m.drawmeridians(np.arange(-180., 181., 1.), labels=[0, 0, 0, 1], fontsize=10)
        m.drawcoastlines()
        m.drawcountries()
        m.drawlsmask(land_color='white', ocean_color='white')
        x_m, y_m = m(lon_o, lat_o)
        av = m.pcolormesh(x_m, y_m, Tt_grid_mean.reshape(nx, ny) - t[tc][ki], cmap='RdBu', vmin=-5, vmax=5)
        x_station_m, y_station_m = m(lon_v, lat_v)
        m.scatter(x_station_m, y_station_m, c='k', marker=".")
        axarr[ki].set_title(datetime[tc][ki][5:-3], fontsize=14)

    cb_ax = fig.add_axes([0.05, 0.1, 0.9, 0.03])
    cbar = fig.colorbar(av, cax=cb_ax, orientation="horizontal")
    fig.tight_layout(rect=(0.01, 0.01, 1, 1))
    cbar.set_label('h')
    plt.savefig(folder_result + '/mapping_mean.png', dpi=200)
    plt.close()

    # Plot mapping (standard deviation)
    lon_m = np.asarray(list(lon_u) + [lon_u[-1] + 0.1, ]) - 0.05
    lat_m = np.array(list(lat_u) + [lat_u[-1] + 0.1, ]) - 0.05
    lon_o, lat_o = np.meshgrid(lon_m, lat_m, indexing='ij')
    fig, axarr = plt.subplots(1, 2 ** I + 1, figsize=(25, 5))
    for ki in range(0, 2 ** I + 1):
        Tt_grid_std = np.std(Tt_grid[ki,:,:], axis=0)

        m = Basemap(llcrnrlon=-3.5, llcrnrlat=4.5, urcrnrlon=0.5, urcrnrlat=8.5, resolution='l', projection='merc',
                    lat_1=20., lat_2=0, lat_0=10., lon_0=5., ax=axarr[ki])
        if ki == 0:
            m.drawparallels(np.arange(-80., 81., 1.), labels=[1, 0, 0, 0], fontsize=10)
        else:
            m.drawparallels(np.arange(-80., 81., 1.), labels=[0, 0, 0, 0], fontsize=10)
        m.drawmeridians(np.arange(-180., 181., 1.), labels=[0, 0, 0, 1], fontsize=10)
        m.drawcoastlines()
        m.drawcountries()
        m.drawlsmask(land_color='white', ocean_color='white')
        x_m, y_m = m(lon_o, lat_o)
        av = m.pcolormesh(x_m, y_m, Tt_grid_std.reshape(nx, ny), cmap='Reds', vmin=0, vmax=0.5)
        x_station_m, y_station_m = m(lon_v, lat_v)
        m.scatter(x_station_m, y_station_m, c='k', marker=".")
        axarr[ki].set_title(datetime[tc][ki][5:-3], fontsize=14)

    cb_ax = fig.add_axes([0.05, 0.1, 0.9, 0.03])
    cbar = fig.colorbar(av, cax=cb_ax, orientation="horizontal")
    fig.tight_layout(rect=(0.01, 0.01, 1, 1))
    cbar.set_label('h')
    plt.savefig(folder_result + '/mapping_std.png', dpi=200)
    plt.close()

print(Tt_grid.shape)
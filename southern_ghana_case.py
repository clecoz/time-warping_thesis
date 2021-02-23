from registration_approach2 import *
from registration_approach1 import *
from registration_approach3 import *

import time
import pyproj
from scipy.spatial.distance import pdist
from itertools import combinations
import os
import pandas as pd
from pykrige import OrdinaryKriging
import math
import matplotlib.pyplot as plt
import matplotlib.colors as cls
from matplotlib.colors import ListedColormap

#=======================================================================================================================
# Southern Ghana case ("All" experiment)
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
# Choose registration parameters

# Choose registration approach
reg = "a3"          # Possible choice: "a1", "a2" and "a3"

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
u = np.zeros(shape=(ntr,ns))
v = np.zeros(shape=(ntr,ns))
for k in range(ns):
    u[:,k] = np.interp(t_resample,t,u_station[:,k])
    v[:,k] = np.interp(t_resample,t,v_station[:,k])


#================================================================
# Define the correlation between stations

# Convert coordinates
proj_latlon = pyproj.Proj(init="epsg:4326")
proj_utm = pyproj.Proj(init="epsg:32630")
x_proj,y_proj = pyproj.transform(proj_latlon, proj_utm, lon_v, lat_v)
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
Acomb = np.zeros((len(list(combinations(range(ns), 2))),ns))
nit = 0
for k, j in combinations(range(ns), 2):
    Acomb[nit,k] = -1
    Acomb[nit, j] = 1
    nit += 1


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
    Tt = registration_a3(u,v,t_resample,I,c1,c2,cs,Acomb,space_corr,folder_result,ks)
else:
    print("Error (wrong ""reg""): this approach does not exist.")


end = time.time()
print('\n Elapsed time for ks {}: {}'.format(ks,end - start))


#================================================================
# Post-processing and Warping
print("\nPost-processing and warping")

# Warped field at the station locations
u_warped_station = mapped(u, t_resample, Tt, I)

# Interpolate the mappings onto the grid
x_proj, y_proj = pyproj.transform(proj_latlon, proj_utm, lon_v, lat_v)
X = (1 / 1000) * np.array([x_proj, y_proj])
xt, yt =  pyproj.transform(proj_latlon, proj_utm, lon_u, lat_u)
xt *= (1/1000)
yt *= (1/1000)
Tt_grid = np.zeros((2**I+1,npts))
if reg == "a1" or reg == "a3":
    for ki in range(2**I+1):
        OK = OrdinaryKriging(X[0,:], X[1,:], Tt[ki,:] , variogram_model='exponential',
                             variogram_parameters={'sill': 1.0, 'range': 150, 'nugget': 0.0}, nlags=50, verbose=False,
                             enable_plotting=False, weight=True, coordinates_type='euclidean')
        z, ss = OK.execute('grid', xt, yt)
        Tt_grid[ki, :] = z.T.reshape(-1)
elif reg == "a2":
    for ks in range(npts):
        Tt_grid[:, ks] = Tt

# Resample the IMERG's field
u_dom_resample = np.zeros(shape=(ntr, npts))
for ks in range(npts):
    u_dom_resample[:,ks] = np.interp(t_resample,t,u_domain.reshape(nt, nx*ny)[:,ks])
u_dom_resample = u_dom_resample.reshape(ntr, nx, ny)

# Warped field at the grid points
u_warped_domain = mapped(u_dom_resample.reshape(ntr,nx*ny), t_resample, Tt_grid, I).reshape(ntr, nx, ny)

#================================================================
# Statistics

if stats:
    print('\nStatistics')
    # The statistics are computed at the station locations where we have gauge measurements

    # Statistics before warping
    RMSE_before = np.sqrt(np.mean((u - v) ** 2))
    MAE_before = np.mean(np.abs(u - v))
    C_before = np.corrcoef(u.reshape(-1), v.reshape(-1))[0, 1]

    # Statistics after warping
    RMSE_after = np.sqrt(np.mean((u_warped_station - v) ** 2))
    MAE_after = np.mean(np.abs(u_warped_station - v))
    C_after = np.corrcoef(u_warped_station.reshape(-1), v.reshape(-1))[0, 1]

    print('#---------------------------------------#')
    print('#        |  RMSE  |  MAE  | Correlation #')
    print('#---------------------------------------#')
    print('# Before |  {:.2f}  |  {:.2f} |    {:.2f}     #'.format(RMSE_before, MAE_before,C_before))
    print('# After  |  {:.2f}  |  {:.2f} |    {:.2f}     #'.format(RMSE_after, MAE_after,C_after))
    print('#---------------------------------------#')


    # ---------------------------------------------
    # Timing error
    # Find indexes of the pixel with the max. rain
    tv = np.argmax(v, axis=0)
    tu = np.argmax(u, axis=0)
    tuw = np.argmax(u_warped_station, axis=0)

    # Compute avrage timing difference
    mask = (~(v < threshold).all(axis=0) * ~(u < 0.1).all(axis=0))
    timing_before = np.mean(np.abs(tu[mask] - tv[mask])) / 10
    timing_after = np.mean(np.abs(tuw[mask] - tv[mask])) / 10

    # Print results
    print("\nAverage timing error for threshold={}mm/h (sample number = {})".format(threshold, np.sum(mask)))
    print("Before: {:.2f} h".format(timing_before))
    print("After:  {:.2f} h".format(timing_after))


    # ---------------------------------------------
    # Position error
    # We need a 2D field to compute position and distance.
    # We use the interpolated stations as truth in order to compute the position error

    # Interpolation of stations onto the 2D grid
    v_domain = np.zeros((nt, nx, ny))
    for k in range(25):
        OK = OrdinaryKriging(lon_v, lat_v, np.sqrt(v_station[k, :]), variogram_model='exponential',
                             variogram_parameters={'sill': 1.0, 'range': 2, 'nugget': 0.01}, nlags=50, verbose=False,
                             enable_plotting=False, weight=True, coordinates_type='geographic')
        z, ss = OK.execute('grid', lon_u, lat_u)
        v_domain[k, :, :] = (z ** 2).T

    # Resampling of the interpolated stations field
    v_dom_resample = np.zeros(shape=(ntr, npts))
    for ks in range(npts):
        v_dom_resample[:, ks] = np.interp(t_resample, t, v_domain.reshape(nt, nx * ny)[:, ks])
    v_dom_resample = v_dom_resample.reshape(ntr, nx, ny)


    # Find indexes of the pixel with the max. rain
    iu, ju = np.unravel_index(np.argmax(u_dom_resample.reshape(ntr, -1), axis=1), (37, 37))
    iv, jv = np.unravel_index(np.argmax(v_dom_resample.reshape(ntr, -1), axis=1), (37, 37))
    iuw, juw = np.unravel_index(np.argmax(u_warped_domain.reshape(ntr, -1), axis=1), (37, 37))


    # Compute distance between max.
    def distance(origin, destination):
        lat1, lon1 = origin
        lat2, lon2 = destination
        radius = 6371  # km
        dlat = math.radians(lat2) - math.radians(lat1)
        dlon = math.radians(lon2) - math.radians(lon1)
        a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
        c = 2 * math.asin(np.sqrt(a))
        return radius * c


    dist_before = np.zeros(ntr)
    dist_after = np.zeros(ntr)
    for kt in range(ntr):
        dist_before[kt] = distance((y[ju[kt]], x[iu[kt]]), (y[jv[kt]], x[iv[kt]]))
        dist_after[kt] = distance((y[juw[kt]], x[iuw[kt]]), (y[jv[kt]], x[iv[kt]]))

    # Compute average position error
    mask = (~(v < threshold).all(axis=1) * ~(u < 0.1).all(axis=1))
    average_dist_before = np.mean(dist_before[mask])
    average_dist_after = np.mean(dist_after[mask])

    # Print results
    print("\nAverage position error for threshold={}mm/h (sample number = {})".format(threshold, np.sum(mask)))
    print("Before: {:.2f} km".format(average_dist_before))
    print("After:  {:.2f} km".format(average_dist_after))


#================================================================
# Plotting

if plot:
    print("\nPlotting")

    # Coordinates
    xx, yy = np.meshgrid(lon_u, lat_u, indexing='ij')
    tc = range(0, nt, int((nt - 1) / 2 ** I))

    # Define colormap
    nws_precip_colors = [
        'white',
        "#04e9e7",  # 0.01 - 0.10 inches
        "#019ff4",  # 0.10 - 0.25 inches
        "#0300f4",  # 0.25 - 0.50 inches
        "#02fd02",  # 0.50 - 0.75 inches
        "#01c501",  # 0.75 - 1.00 inches
        "#008e00",  # 1.00 - 1.50 inches
        "#fdf802",  # 1.50 - 2.00 inches
        "#e5bc00",  # 2.00 - 2.50 inches
        "#fd9500",  # 2.50 - 3.00 inches
        "#fd0000",  # 3.00 - 4.00 inches
        "#d40000",  # 4.00 - 5.00 inches
        "#bc0000",  # 5.00 - 6.00 inches
        "#f800fd",  # 6.00 - 8.00 inches
        "#9854c6",  # 8.00 - 10.00 inches
        "#fdfdfd"  # 10.00+
    ]
    precip_colormap = cls.ListedColormap(nws_precip_colors)
    clevs = [0, 0.02, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40, 50, 70, 100]  # , 150, 200, 250, 300, 400, 500, 600, 750]
    norm = cls.BoundaryNorm(clevs, 13)
    my_cmap = precip_colormap(np.arange(precip_colormap.N))
    my_cmap = ListedColormap(my_cmap)

    # Plot the 2D  rainfall fields before warping (u and v in background and contour respectively)
    lon_m = np.asarray(list(lon_u) + [lon_u[-1] + 0.1, ]) - 0.05
    lat_m = np.array(list(lat_u) + [lat_u[-1] + 0.1, ]) - 0.05
    lon_o, lat_o = np.meshgrid(lon_m, lat_m, indexing='ij')
    fig, axarr = plt.subplots(4, 6, figsize=(30, 15))
    pos = 0
    for kt in range(0, nt - 1):
        if kt == 12:
            av = axarr[int(pos / 6), pos % 6].pcolormesh(lon_o, lat_o, u_domain[kt, :, :], cmap=my_cmap, norm=norm)
        else:
            axarr[int(pos / 6), pos % 6].pcolormesh(lon_o, lat_o, u_domain[kt, :, :], cmap=my_cmap, norm=norm)
        axarr[int(pos / 6), pos % 6].contour(xx, yy, v_domain[kt, :, :], cmap=my_cmap, norm=norm)
        axarr[int(pos / 6), pos % 6].scatter(lon_v, lat_v, c=v_station[kt, :], s=40, cmap=precip_colormap,
                                             edgecolor='black',
                                             norm=norm)
        axarr[int(pos / 6), pos % 6].set(adjustable='box-forced', aspect='equal')
        axarr[int(pos / 6), pos % 6].set_title("Time {}".format(kt))
        pos += 1
    plt.tight_layout()
    cbar = fig.colorbar(av, ax=axarr)
    cbar.set_label('mm/h')
    plt.savefig(folder_result + '/input_fields.png', dpi=200)
    plt.close()

    # Plot the warped field and the "true" rainfall (background and contour respectively)
    lon_m = np.asarray(list(lon_u) + [lon_u[-1] + 0.1, ]) - 0.05
    lat_m = np.array(list(lat_u) + [lat_u[-1] + 0.1, ]) - 0.05
    lon_o, lat_o = np.meshgrid(lon_m, lat_m, indexing='ij')
    fig, axarr = plt.subplots(4, 6, figsize=(30, 15))
    pos = 0
    for ktr in range(0,ntr-1,10):
        kt = int(ktr/10)
        if kt == 12:
            av = axarr[int(pos / 6), pos % 6].pcolormesh(lon_o, lat_o, u_warped_domain[ktr, :, :], cmap=my_cmap, norm=norm)
        else:
            axarr[int(pos / 6), pos % 6].pcolormesh(lon_o, lat_o, u_warped_domain[ktr, :, :], cmap=my_cmap, norm=norm)
        axarr[int(pos / 6), pos % 6].contour(xx, yy, v_dom_resample[ktr, :, :], cmap=my_cmap, norm=norm)
        axarr[int(pos / 6), pos % 6].scatter(lon_v, lat_v, c=v_station[kt, :], s=40, cmap=precip_colormap,
                                             edgecolor='black',norm=norm)
        axarr[int(pos / 6), pos % 6].set(adjustable='box-forced', aspect='equal')
        axarr[int(pos / 6), pos % 6].set_title("Time {}".format(kt))
        pos += 1
    plt.tight_layout()
    cbar = fig.colorbar(av, ax=axarr)
    cbar.set_label('mm/h')
    plt.savefig(folder_result + '/warped_field.png', dpi=200)
    plt.close()


    # Plot mapping
    lon_m = np.asarray(list(lon_u) + [lon_u[-1] + 0.1, ]) - 0.05
    lat_m = np.array(list(lat_u) + [lat_u[-1] + 0.1, ]) - 0.05
    lon_o, lat_o = np.meshgrid(lon_m, lat_m, indexing='ij')
    fig, axarr = plt.subplots(1,2**I+1, figsize=(20, 5))
    for ki in range(0, 2**I+1):
        av = axarr[ki].pcolormesh(lon_o, lat_o, Tt_grid[ki, :].reshape(nx, ny) - t[tc][ki], cmap='RdBu', vmin=-5, vmax=5)
        if reg == "a2":
            axarr[ki].scatter(lon_v, lat_v, c=(Tt[ki] - t[tc][ki])*np.ones(ns), s=50, edgecolor='black', cmap='RdBu', vmin=-5, vmax=5)
        else:
            axarr[ki].scatter(lon_v, lat_v, c=Tt[ki, :] - t[tc][ki], s=50, edgecolor='black', cmap='RdBu', vmin=-5, vmax=5)
        axarr[ki].set(adjustable='box-forced', aspect='equal')
        axarr[ki].set_xlim(np.min(lon_o), np.max(lon_o))
        axarr[ki].set_ylim(np.min(lat_o), np.max(lat_o))
        axarr[ki].set_title("Time {}".format(int(t[tc][ki])))

    cb_ax = fig.add_axes([0.05, 0.1, 0.9, 0.03])
    cbar = fig.colorbar(av, cax=cb_ax, orientation="horizontal")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    cbar.set_label('h')
    plt.savefig(folder_result + '/mapping.png', dpi=200)
    plt.close()


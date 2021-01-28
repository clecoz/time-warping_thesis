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

import time
#===========================================================================

def registration_a3(u,v,t,I,c1,c2,c3,Acomb,space_corr,folder_results,ks):

    nt = len(t)
    ns = u.shape[1]

    # Iteration on the morphing grids (i<=I)
    for i in range(1, I + 1):
        print('\n Step' + str(i))
        start_time = time.time()
        mi = 2 ** i + 1

        # Initialization
        if i == 1:
            tc = range(0, nt, int((nt - 1) / 2 ** i))
            tT2 = np.zeros((mi, ns))
            for k in range(ns):
                tT2[:, k] = t[tc]
        else:
            tT2 = np.zeros((mi, ns))
            tc_new = np.linspace(0, nt - 1, mi, dtype=int)
            for k in range(ns):
                T = interpolate.interp1d(t[tc], tT[:, k])
                tT2[:, k] = T(t[tc_new])
            tc = tc_new
        tT = tT2.reshape(-1)

        # Smooth signals
        vs = smooth(v, t, i)
        us = smooth(u, t, i)

        # Normalize ?
        for k in range(ns):
            if np.max(us[:,k]) > 0:
                us[:,k] = np.max(vs[:,k]) * us[:,k] / np.max(us[:,k])


        # Define constrains
        cons = {'type': 'ineq', 'fun': constr1_bis, 'args': (t, i,ns)}
        cons2 = {'type': 'ineq', 'fun': constr2_bis, 'args': (t, i,ns)}
        cons3 = {'type': 'ineq', 'fun': constr3_bis, 'args': (t, i,ns)}

        # Optimize
        tTo = op.minimize(J_der, tT, args=(us, vs, t, i, c1, c2, c3,Acomb,space_corr), jac=True, constraints=[cons3, cons2, cons], method='SLSQP', options={'maxiter': 10000})
        print(tTo.message)

        # Update mapping
        tT = tTo.x.reshape((mi,ns))

        # Save mapping for future use
        np.savetxt(folder_results + '/mtT_i{}_ks{}.csv'.format(i,ks), tT, delimiter=',')

        # print elapsed time
        elapsed_time = time.time() - start_time
        print('Elapsed time: {}'.format(elapsed_time))

    # Return mapping
    return tT



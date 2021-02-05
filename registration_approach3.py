import scipy.optimize as op
from morphing_function import *

#===========================================================================

def registration_a3(u,v,t,I,c1,c2,cs,Acomb,space_corr,folder_results,ks):
    # This function perform the automatic registration for the time warping and return the mappings.
    # It takes as input:
    # - u, the rainfall time series to be corrected.
    # - v, the target rainfall time series (assumed to be the truth). The inputs u and v need to have the same dimensions.
    # - t, the time coordinates.
    # - the number of steps I (corresponding to the number of morphing grid). I has to be an integer.
    # - the regulation coefficients c1, c2 and cs (floats).
    # - Acomb is a matrix pariring two by two the stations and space_corr the corresponding correlation. Together, they define the influence function.
    # - folder_results is a folder, the mappings will be saved in this folder for future use.
    # - ks is added to the file names containing the mappings (used for the LOOV experiment).

    nt = len(t)
    ns = u.shape[1]

    # Iteration on the morphing grids (i<=I)
    for i in range(1, I + 1):
        print('\n Step' + str(i))
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

        # Normalize
        for k in range(ns):
            if np.max(us[:,k]) > 0:
                us[:,k] = np.max(vs[:,k]) * us[:,k] / np.max(us[:,k])


        # Define constrains
        cons = {'type': 'ineq', 'fun': constr1_bis, 'args': (t, i,ns)}
        cons2 = {'type': 'ineq', 'fun': constr2_bis, 'args': (t, i,ns)}
        cons3 = {'type': 'ineq', 'fun': constr3_bis, 'args': (t, i,ns)}

        # Optimize
        tTo = op.minimize(J_a3, tT, args=(us, vs, t, i, c1, c2, cs,Acomb,space_corr), jac=True, constraints=[cons3, cons2, cons], method='SLSQP', options={'maxiter': 10000})
        print(tTo.message)

        # Update mapping
        tT = tTo.x.reshape((mi,ns))

        # Save mapping for future use
        np.savetxt(folder_results + '/mtT_i{}_ks{}.csv'.format(i,ks), tT, delimiter=',')


    # Return mapping
    return tT



import scipy.optimize as op
from morphing_function import *

#===========================================================================

def registration_a2(u,v,t,I,c1,c2,folder_results,ks):
    # This function performs the automatic registration for the time warping and return the mappings.
    # It takes as input:
    # - u, the rainfall time series to be corrected.
    # - v, the target rainfall time series (assumed to be the truth). The inputs u and v need to have the same dimensions.
    # - t, the time coordinates.
    # - the number of steps I (corresponding to the number of morphing grid). I has to be an integer.
    # - the regulation coefficients c1 and c2 (floats).
    # - folder_results is a folder, the mappings will be saved in this folder for future use.
    # - ks is added to the file names containing the mappings (used for the LOOV experiment).

    nt = len(t)
    ns = u.shape[1]

    # Iteration on the morphing grids (i<=I)
    for i in range(1, I + 1):
        print('Step i=' + str(i))

        # Initialization
        if i == 1:
            tc = range(0, nt, int((nt - 1) / 2 ** i))
            tT = t[tc]
        else:
            T = interpolate.interp1d(t[tc], tT)
            tc = np.linspace(0, nt - 1, (2 ** i + 1), dtype=int)
            tT = T(t[tc])

        # Smooth signals
        vs = smooth(v, t, i)
        us = smooth(u, t, i)

        # Normalize
        for k in range(ns):
            if np.max(us[:,k]) > 0:
                us[:,k] = np.max(vs[:,k]) * us[:,k] / np.max(us[:,k])

        # Define constrains
        cons = {'type': 'ineq', 'fun': constr1_bis, 'args': (t, i,1)}
        cons2 = {'type': 'ineq', 'fun': constr2_bis, 'args': (t, i,1)}
        cons3 = {'type': 'ineq', 'fun': constr3_bis, 'args': (t, i,1)}

        # Optimize
        tTo = op.minimize(J_a1_a2, tT, args=(us, vs, t, i, c1, c2), constraints=[cons3, cons2, cons], method='SLSQP')
        print(tTo.success)
        print(tTo.message)

        # Update mapping
        tT = tTo.x

        # Save mapping for future use
        np.savetxt(folder_results + '/mtT_i{}_ks{}.csv'.format(i,ks), tT, delimiter=',')

    # Return mapping
    return tT
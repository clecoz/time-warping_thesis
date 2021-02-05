import scipy.optimize as op
from morphing_function import *

#===========================================================================

def registration_a1(u,v,t,I,c1,c2,folder_results,ks):
    # This function perform the automatic registration for the time warping and return the mappings.
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

    if I == 1:
        tT_res1 = np.zeros((2 ** I + 1, ns))
    elif I == 2:
        tT_res1 = np.zeros((2 ** 1 + 1, ns))
        tT_res2 = np.zeros((2 ** I + 1, ns))
    elif I == 3:
        tT_res1 = np.zeros((2 ** 1 + 1, ns))
        tT_res2 = np.zeros((2 ** 2 + 1, ns))
        tT_res3 = np.zeros((2 ** I + 1, ns))
    elif I == 4:
        tT_res1 = np.zeros((2 ** 1 + 1, ns))
        tT_res2 = np.zeros((2 ** 2 + 1, ns))
        tT_res3 = np.zeros((2 ** 3 + 1, ns))
        tT_res4 = np.zeros((2 ** I + 1, ns))

    tT_return = np.zeros((2 ** I + 1, ns))


    # Loop on the stations
    for k in range(ns):
        print('Station {}'.format(k))

        # Iteration on the morphing grids (i<=I)
        for i in range(1, I + 1):
            # Initialization
            if i == 1:
                tc = range(0, nt, int((nt - 1) / 2 ** i))
                tT = t[tc]
            else:
                T = interpolate.interp1d(t[tc], tT)
                tc = np.linspace(0, nt - 1, (2 ** i + 1), dtype=int)
                tT = T(t[tc])

            # Smooth signals
            vs = np.expand_dims(smooth(v[:, k], t, i), axis=1)
            us = np.expand_dims(smooth(u[:, k], t, i), axis=1)

            # Normalize
            if np.max(us) != 0:
                us = np.max(np.max(vs)) * us / np.max(np.max(us))

            # Define constrains
            cons = {'type': 'ineq', 'fun': constr1_bis, 'args': (t, i, 1)}
            cons2 = {'type': 'ineq', 'fun': constr2_bis, 'args': (t, i, 1)}
            cons3 = {'type': 'ineq', 'fun': constr3_bis, 'args': (t, i, 1)}

            # Optimize
            tTo = op.minimize(J_a1_a2, tT, args=(us, vs, t, i, c1, c2), constraints=[cons3, cons2, cons], method='SLSQP')
            print(tTo.message)

            # Update mapping
            tT = tTo.x

            # Store results
            if i == 1:
                tT_res1[:,k] = tT
            elif i== 2:
                tT_res2[:,k] = tT
            elif i == 3:
                tT_res3[:,k] = tT
            elif i == 4:
                tT_res4[:,k] = tT

            if i == I:
                tT_return[:, k] = tT

    # Save mapping for future use (only for I<5)
    if I == 1:
        np.savetxt(folder_results + '/mtT_i1_ks{}.csv'.format(ks), tT_res1, delimiter=',')
    elif I == 2:
        np.savetxt(folder_results + '/mtT_i1_ks{}.csv'.format(ks), tT_res1, delimiter=',')
        np.savetxt(folder_results + '/mtT_i2_ks{}.csv'.format(ks), tT_res2, delimiter=',')
    elif I == 3:
        np.savetxt(folder_results + '/mtT_i1_ks{}.csv'.format(ks), tT_res1, delimiter=',')
        np.savetxt(folder_results + '/mtT_i2_ks{}.csv'.format(ks), tT_res2, delimiter=',')
        np.savetxt(folder_results + '/mtT_i3_ks{}.csv'.format(ks), tT_res3, delimiter=',')
    elif I == 4:
        np.savetxt(folder_results + '/mtT_i1_ks{}.csv'.format(ks), tT_res1, delimiter=',')
        np.savetxt(folder_results + '/mtT_i2_ks{}.csv'.format(ks), tT_res2, delimiter=',')
        np.savetxt(folder_results + '/mtT_i3_ks{}.csv'.format(ks), tT_res3, delimiter=',')
        np.savetxt(folder_results + '/mtT_i4_ks{}.csv'.format(ks), tT_res4, delimiter=',')


    return tT_return

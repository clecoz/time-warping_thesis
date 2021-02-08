import numpy as np
from scipy import interpolate
from interpolation2 import interpn_linear
from scipy.sparse import csr_matrix, diags, issparse

########################################################################################################################
#
# Cost function and related functions
#
########################################################################################################################

def smooth(v,t,i):
    # This function returns the smoothed signal
    # It takes as inputs:
    # - the input signal v (time series) to be smoothed
    # - the corresponding time coordinate t
    # - the step number i which defines the level of smoothing

    v1 = np.zeros(v.shape)
    nt = len(t)
    alpha = 0.1 / (2 ** (i*2) + 1)
    for j in np.arange(0, nt):
        j = int(j)
        tloc = t[j]
        kernel_t = np.exp(-((t - tloc) / 25) ** 2 / alpha) / sum(np.exp(-((t - t[int((nt - 1) / 2)]) /25)** 2 / alpha))
        if len(v.shape) == 1:
            v1[j] = sum(v * kernel_t)
        elif len(v.shape) == 2:
            for k in range(v.shape[1]):
                v1[j,k] = sum(v[:,k] * kernel_t)
    return v1


#-----------------------------------------------------------------------------------
# Warping functions
def mapped(u,t,tT,i):
    # This function returns the warped signal
    # It takes as inputs:
    # - the input signal u (time series)
    # - the corresponding time coordinate t
    # - the mapping tT
    # - the corresponding step number i (which is linked to the resolution of the mapping)

    nt = len(t)
    tc = np.linspace(0, nt - 1, (2 ** i + 1), dtype=int)

    if len(tT.shape)==1:
        # Transform coordinate
        t_prime = interpolate.griddata(t[tc], tT, t)
        # Interpolated function
        if len(u.shape) == 1:
            uf = interpolate.interp1d(t, u, fill_value=0,bounds_error=False) #"extrapolate")
            umap = uf(t_prime)
        elif len(u.shape) == 2:
            umap = np.zeros(u.shape)
            for j in range(u.shape[1]):
                uf = interpolate.interp1d(t, u[:, j], fill_value="extrapolate",bounds_error=False) #"extrapolate")
                umap[:, j] = uf(t_prime)

    elif len(tT.shape)==2:
        umap = np.zeros(u.shape)
        for j in range(u.shape[1]):
            # Transform coordinate
            t_prime = interpolate.griddata(t[tc], tT[:,j], t)
            # Interpolated function
            umap[:, j] = interpolate.interpn((t,), u[:,j], t_prime, method='linear', bounds_error=False, fill_value=0)

    return umap


def mapped_weight(u, t, tT, i):
    # This function returns the warped signal and the corresponding interpolation weight (used in the computation of the derivative of the cost function).
    # It takes as inputs:
    # - the input signal u (time series)
    # - the corresponding time coordinate t
    # - the mapping tT
    # - the corresponding step number i (which is linked to the resolution of the mapping)

    nt = len(t)
    tc = np.linspace(0, nt - 1, (2 ** i + 1), dtype=int)

    uT = np.zeros(u.shape)
    uT_t = np.zeros(u.shape)
    for j in range(u.shape[1]):
        # Transform coordinate
        t_prime = interpolate.griddata(t[tc], tT[:, j], t)
        # Interpolate
        uT[:, j], uT_t[:, j] = interpn_linear((t,), u[:, j], np.expand_dims(t_prime,axis=1), method='linear', bounds_error=False, fill_value=0)

    return uT, uT_t


#-----------------------------------------------------------------------------------
# Derative function (used in cost_function)

def dXdT(t,i):
    nt = len(t)
    mi = 2**i + 1
    dnt = int((nt-1) / 2 ** i)

    dtdT2 = np.zeros((nt,mi))
    for k in range(1,mi-1):
        A1 = (t[dnt*(k+1)] - t)/(t[dnt*(k+1)]-t[dnt*(k)]) * (t <= t[dnt * (k + 1)]) * (t > t[dnt * k])
        A2 = (t - t[dnt*(k-1)])/(t[dnt*(k)]-t[dnt*(k-1)]) * (t <= t[dnt * k]) * (t > t[dnt * (k - 1)])
        dtdT2[:,k] = A1 + A2

    k=0
    A1 = (t[dnt * (k + 1)] - t) / (t[dnt * (k + 1)] - t[dnt * (k)]) * (t <= t[dnt * (k + 1)]) * (t >= t[dnt * k])
    dtdT2[:, k] = A1

    k=mi-1
    A2 = (t - t[dnt * (k - 1)]) / (t[dnt * (k)] - t[dnt * (k - 1)]) * (t <= t[dnt * k]) * (t > t[dnt * (k - 1)])
    dtdT2[:, k] = A2

    return dtdT2


#-----------------------------------------------------------------------------------
# Constraint functions

def constr1(tTr,t,i,ns):
    # Check if first constraint is respected
    mi = 2 ** i + 1
    tT = tTr.reshape((mi, ns))
    nt = len(t)
    tc = np.linspace(0, nt - 1, (2 ** i + 1), dtype=int)
    t_prime = np.zeros((len(t),ns))
    for j in range(ns):
        t_prime[:, j] = interpolate.griddata(t[tc], tT[:, j], t)
    return np.min(t_prime[1:-1,:]-t_prime[0:-2,:])

def constr2(tTi,t,i,ns):
    # Check if second constraint is respected
    nt = len(t)
    mi = 2 ** i + 1
    tT = tTi.reshape((mi, ns))
    tc = np.linspace(0, nt - 1, (2 ** i + 1), dtype=int)
    t_prime = np.zeros((len(t), ns))
    for j in range(ns):
        t_prime[:, j] = interpolate.griddata(t[tc], tT[:, j], t)
    return max(t) - np.max(t_prime)


def constr3(tTi,t,i,ns):
    # Check if third constraint is respected
    nt = len(t)
    mi = 2 ** i + 1
    tT = tTi.reshape((mi, ns))
    tc = np.linspace(0, nt - 1, (2 ** i + 1), dtype=int)
    t_prime = np.zeros((len(t), ns))
    for j in range(ns):
        t_prime[:, j] = interpolate.griddata(t[tc], tT[:, j], t)
    return np.min(t) - np.min(tTi)



def constr1_bis(tTr,t,i,ns):
    # Check if the first constraint is respected
    mi = 2 ** i + 1
    tT = tTr.reshape((mi, ns))
    return np.min(tT[1:-1,:]-tT[0:-2,:],axis=0)

def constr2_bis(tTi,t,i,ns):
    # Check if the second constraint is respected
    mi = 2 ** i + 1
    tT = tTi.reshape((mi, ns))
    return tT[-1,:] - np.max(t)

def constr3_bis(tTi,t,i,ns):
    # Check if the third constraint is respected
    mi = 2 ** i + 1
    tT = tTi.reshape((mi, ns))
    return np.min(t) - tT[0,:]




#-----------------------------------------------------------------------------------
# Cost functions

def J_a3(tTr,us,vs,t,i,c1,c2,cs,Acomb=None,space_corr=None):
    # This function returns the value and the derivative of the cost function for the approach A3
    # It takes as input:
    # - the mapping tTr.
    # - the smoothed inputs us and vs.
    # - the time coordinates t.
    # - the step i (defining the smoothing and the resolution of the mapping)
    # - the regulation coefficients c1, c2 and cs
    # For approach A3: Acomb is a matrix pairing two by two the stations and space_corr the corresponding correlation. Together, they define the influence function.

    mi = 2**i + 1
    ns = us.shape[1]
    tT = tTr.reshape((mi,ns))

    nt = len(t)
    tc = np.linspace(0,nt-1,(2**i+1),dtype=int)
    T = np.zeros(tT.shape)
    for k in range(ns):
        T[:,k] = t[tc]
    dT = t[tc[1]] - t[tc[0]]

    v1 = vs
    u1 = mapped(us,t,tT,i)

    Tdif = tT - T

    #--------------------------------------------------
    # Cost
    At = np.diag(1 * np.ones(mi - 1), 1) + np.diag(-1 * np.ones(mi ), 0)
    At[mi-1,mi-1] = -1
    At /= dT
    At = csr_matrix(At)

    err = (v1 - u1).reshape(-1)
    Jo = np.sqrt(err@err)
    tdif = Tdif.reshape(-1)
    dTdif = At @ Tdif
    dtdif = dTdif.reshape(-1)
    Jb = c1 * np.sqrt(tdif@tdif/mi)   + c2 * np.sqrt(dtdif@dtdif/mi)

    if (Acomb is not None) and (cs!=0):
        Tdif_s = Acomb @ Tdif.T
        C = diags(space_corr)
        Tdif_sc = (C @ Tdif_s).reshape(-1)
        Js = cs * np.sqrt(Tdif_sc.T @ Tdif_sc /mi)
    else:
        Js = 0

    cost = Jo + Jb + Js

    # --------------------------------------------------
    # Derivative
    _, uT_t = mapped_weight(us,t,tT,i)
    dt = round(t[1]-t[0],2)
    dtdT = dXdT(t,i)

    if (err @ err)==0:
        jac = np.zeros(mi)
    else:
        jac = - ((err @ err) ** (-1 / 2)) * (((v1 - u1) * (uT_t / dt)).T @ dtdT)

    if (tdif@tdif)==0 or c1==0:
        jac1 = 0
    else:
        jac1 = (c1 * (mi  ) ** (-1 / 2) * (tdif @ tdif) ** (-1 / 2) * Tdif).T

    if (dtdif@dtdif)==0 or c2==0:
        jac2 = 0
    else:
        jac2 = (c2 * (mi ) ** (-1 / 2) * (dtdif @ dtdif) ** (-1 / 2) * At.T @ dTdif).T

    if (Acomb is not None) and (cs!=0):
        if (Tdif_sc.T @ Tdif_sc)==0 or cs==0:
            ja  = 0
        else:
            jac3 = cs * (mi ) ** (-1 / 2) * (Tdif_sc.T @ Tdif_sc) ** (-1 / 2) *  Acomb.T @ C.T  @ C @ Tdif_s
    else:
        jac3 = 0

    jac = jac + jac1 + jac2 + jac3

    return cost, jac.T.reshape(-1)



def J_a1_a2(tTr,us,vs,t,i,c1,c2):
    # This function returns the value of the cost function for the approaches A1 and A2
    # It takes as input:
    # - the mapping tTr.
    # - the smoothed inputs us and vs.
    # - the time coordinates t.
    # - the step i (defining the smoothing and the resolution of the mapping)
    # - the regulation coefficients c1 and c2

    mi = 2**i + 1

    nt = len(t)
    if len(us.shape) == 1:
        ns = 1
    elif len(us.shape) == 2:
        ns = us.shape[1]

    tT = tTr
    tc = np.linspace(0,nt-1,(2**i+1),dtype=int)

    v1 = vs
    u1 = mapped(us,t,tT,i)

    dT = t[tc[1]] - t[tc[0]]
    Tdif = tT - t[tc]

    err = (v1 - u1).reshape(-1)
    Jo = np.sqrt(err @ err)
    tdif = Tdif.reshape(-1)
    At = np.diag(1 * np.ones(mi - 1), 1) + np.diag(-1 * np.ones(mi), 0)
    At[mi - 1, mi - 1] = -1
    At /= dT
    dTdif = At @ Tdif
    dtdif = dTdif.reshape(-1)
    Jb = c1 * np.sqrt(tdif @ tdif / mi ) \
         + c2 * np.sqrt(dtdif @ dtdif / mi )

    cost = Jo + Jb
    return cost


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
from interpolation2 import interpn_linear

from scipy.sparse import csr_matrix, diags, issparse

########################################################################################################################
def smooth(v,t,i):
    # Return smoothed signal
    v1 = np.zeros(v.shape)
    #print(len(v.shape))
    #print(v1.shape)
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


def mapped(u,t,tT,i):
    # Return warped signal
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
                #umap[:,j] = np.squeeze(interpolate.interpn((t,), u[:,j], t_prime, method='linear', bounds_error=False,fill_value=0))
                uf = interpolate.interp1d(t, u[:, j], fill_value="extrapolate",bounds_error=False) #"extrapolate")
                umap[:, j] = uf(t_prime)

    elif len(tT.shape)==2:
        umap = np.zeros(u.shape)
        for j in range(u.shape[1]):
            # Transform coordinate
            t_prime = interpolate.griddata(t[tc], tT[:,j], t)
            # Interpolated function
            umap[:, j] = interpolate.interpn((t,), u[:,j], t_prime, method='linear', bounds_error=False, fill_value=0)

            #uf = interpolate.interp1d(t,u[:,j],fill_value=0,bounds_error=False) #"extrapolate")
            #umap[:,j] = uf(t_prime)

    return umap


def mapped_weight(u, t, tT, i):
    # Return warped signal
    nt = len(t)
    tc = np.linspace(0, nt - 1, (2 ** i + 1), dtype=int)

    uT = np.zeros(u.shape)
    uT_t = np.zeros(u.shape)
    for j in range(u.shape[1]):
        # Transform coordinate
        t_prime = interpolate.griddata(t[tc], tT[:, j], t)
        #print(t_prime)
        uT[:, j], uT_t[:, j] = interpn_linear((t,), u[:, j], np.expand_dims(t_prime,axis=1), method='linear', bounds_error=False, fill_value=0)

        #dt = np.zeros(t_prime.shape)
        #dt[1::] = t_prime[1::] - t_prime[0:-1:]
        #dt[0] = dt[1]-dt[0]
        #uT_t[:, j] = uT_t[:, j]  *dt

    return uT, uT_t


def dXdT(t,i):
    nt = len(t)
    mi = 2**i + 1
    dnt = int((nt-1) / 2 ** i)
    #print(dnt)
    #print(mi)
    #print(nt)

    dtdT2 = np.zeros((nt,mi))
    for k in range(1,mi-1):
        #print(k)
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



#=============================
def constr1(tTr,t,i,ns):
    #var = tT
    mi = 2 ** i + 1
    tT = tTr.reshape((mi, ns))
    nt = len(t)
    tc = np.linspace(0, nt - 1, (2 ** i + 1), dtype=int)
    t_prime = np.zeros((len(t),ns))
    #T = np.zeros((len(t), ns))
    for j in range(ns):
        t_prime[:, j] = interpolate.griddata(t[tc], tT[:, j], t)
    #    T[:, j] = t
    #t_prime = interpolate.griddata(t[tc], tT, t)
    #print( min(t_prime[1:-1]-t_prime[0:-2]))
    #print(np.min(t_prime[1:-1,:]-t_prime[0:-2,:]) )
    return np.min(t_prime[1:-1,:]-t_prime[0:-2,:]) #min(var[1:-1]-var[0:-2])

def constr2(tTi,t,i,ns):
    nt = len(t)
    mi = 2 ** i + 1
    tT = tTi.reshape((mi, ns))
    tc = np.linspace(0, nt - 1, (2 ** i + 1), dtype=int)
    t_prime = np.zeros((len(t), ns))
    for j in range(ns):
        t_prime[:, j] = interpolate.griddata(t[tc], tT[:, j], t)
    #t_prime = interpolate.griddata(t[tc], tTi, t)
    return max(t) - np.max(t_prime) #max(t) - max(tTi) #
    #return max(t) - t_prime[-1]

def constr3(tTi,t,i,ns):
    nt = len(t)
    mi = 2 ** i + 1
    tT = tTi.reshape((mi, ns))
    tc = np.linspace(0, nt - 1, (2 ** i + 1), dtype=int)
    t_prime = np.zeros((len(t), ns))
    for j in range(ns):
        t_prime[:, j] = interpolate.griddata(t[tc], tT[:, j], t)
    # t_prime = interpolate.griddata(t[tc], tTi, t)
    #return np.min(t_prime)-min(t) #min(tTi)-min(t) #
    return np.min(t) - np.min(tTi)
    #return t_prime[0] - min(t)


def constr1_bis(tTr,t,i,ns):
    mi = 2 ** i + 1
    tT = tTr.reshape((mi, ns))
    #if np.min(tT[1:-1,:]-tT[0:-2,:])<0:
    #    print(tT)
    #    print(np.min(tT[1:-1,:]-tT[0:-2,:]))
    #    print(tT[1:-1,:]-tT[0:-2,:])
    #    plt.close()
    #    plt.plot(tT)
    #    plt.show()
    #    plt.plot(tT[1:-1,:]-tT[0:-2,:])
    #    plt.show()
        #exit()
    #print(np.min(tT[1:-1,:]-tT[0:-2,:],axis=0))
    #exit()
    return np.min(tT[1:-1,:]-tT[0:-2,:],axis=0)

def constr2_bis(tTi,t,i,ns):
    mi = 2 ** i + 1
    tT = tTi.reshape((mi, ns))
    return tT[-1,:] - np.max(t)

def constr3_bis(tTi,t,i,ns):
    mi = 2 ** i + 1
    tT = tTi.reshape((mi, ns))
    return np.min(t) - tT[0,:]

#=============================
# Cost function
def J(tTr,us,vs,t,i,c1,c2,c3,Acomb=None,space_corr=None):
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

    uT, uT_t = mapped_weight(us,t,tT,i)
    #print(uT.shape)
    #print(uT_t.shape)

    #Tdif = np.zeros(tT.shape)
    #for k in range(ns):
    #    Tdif[:,k] = tT[:,k] - t[tc]
    Tdif = tT - T

    # Cost
    #print(u1.T)
    #plt.plot(v1,'+', label='v1')
    #plt.plot(u1,'x', label='u1')
    #plt.plot(us,'.', label='us')
    #plt.legend()
    #plt.show()
    #Jo = np.sqrt(np.sum((v1-u1)**2))
    #Jb = c1 * np.sqrt(np.sum(Tdif**2)/mi) \
    #    + c2 * np.sqrt(np.sum(((Tdif[0:mi-1,:]-Tdif[1:mi,:])/(T[0:mi-1,:]-T[1:mi,:]))**2)/mi)
    #Jb = c1 * np.sum(np.sqrt(np.sum(Tdif**2,axis=0)/mi))/ns \
    #    + c2 * np.sum(np.sqrt(np.sum(((Tdif[0:mi-1,:]-Tdif[1:mi,:])/(T[0:mi-1,:]-T[1:mi,:]))**2, axis=0)/mi))/ns

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

    #print('Jo ' + str(Jo))
    #print(tT)


    if Acomb is not None:
        #print(Tdif.shape)
        #print(Acomb.shape)
        Tdif_s = Acomb @ Tdif.T
        C = np.diag(np.sqrt(space_corr))

        Tdif_sc = (C @ Tdif_s).reshape(-1)
        #print(Tdif_sc.shape)
        #print(C.shape)
        #print(Tdif_sc.T @ Tdif_sc)
        Js = c3 * np.sqrt(Tdif_sc.T @ Tdif_sc/mi)
    else:
        Js = 0


    #cost = np.sqrt(sum((v1-u1)**2)) + 0.1/mi*np.sqrt(sum((tT-t[xc])**2)) + 0.1/mi*np.sqrt(sum(((Tdif[0:mi-1]-Tdif[1:mi])/(t[xc[0:mi-1]]-t[xc[1:mi]]))**2))
    cost = Jo + Jb + Js

    return cost


#=============================
# Cost function with derivative
def J_der(tTr,us,vs,t,i,c1,c2,c3,Acomb=None,space_corr=None):
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

    #Tdif = (tT - T).reshape(-1)  #tT - T
    Tdif = tT - T


    # Cost
    At = np.diag(1 * np.ones(mi - 1), 1) + np.diag(-1 * np.ones(mi ), 0)
    At[mi-1,mi-1] = -1
    At /= dT
    At = csr_matrix(At)

    err = (v1 - u1).reshape(-1)
    Jo = np.sqrt(err@err) #np.sqrt(np.sum((v1-u1)**2))
    tdif = Tdif.reshape(-1)
    dTdif = At @ Tdif
    dtdif = dTdif.reshape(-1)
    Jb = c1 * np.sqrt(tdif@tdif/mi) \
         + c2 * np.sqrt(dtdif@dtdif/mi)

    if (Acomb is not None) and (c3!=0):
        Tdif_s = Acomb @ Tdif.T
        C = diags(space_corr)
        #C = np.diag(np.sqrt(space_corr))
        #C = np.diag(space_corr)
        #Tdif_sc = csr_matrix((C @ Tdif_s).reshape(-1)).T
        Tdif_sc = (C @ Tdif_s).reshape(-1)
        Js = c3 * np.sqrt(Tdif_sc.T @ Tdif_sc /mi)
    else:
        Js = 0

    cost = Jo + Jb + Js

    # Derivative
    _, uT_t = mapped_weight(us,t,tT,i)
    dt = round(t[1]-t[0],2)
    dtdT = dXdT(t,i)

    #jac = - ((err@err) ** (-1 / 2)) * np.sum((((v1 - u1) * (uT_t/dt)).T @ dtdT),axis=0)

    if (err @ err)==0:
        jac = np.zeros(mi)
    else:
        jac = - ((err @ err) ** (-1 / 2)) * (((v1 - u1) * (uT_t / dt)).T @ dtdT)

    if (tdif@tdif)==0 or c1==0:
        jac1 = 0
    else:
        #jac1 = c1 * (mi*ns)**(-1/2) * (tdif@tdif)**(-1/2) * np.sum(Tdif,axis=1)
        jac1 = (c1 * (mi  ) ** (-1 / 2) * (tdif @ tdif) ** (-1 / 2) * Tdif).T


    if (dtdif@dtdif)==0 or c2==0:
        jac2 = 0
    else:
        #jac2 = c2 * (mi*ns)**(-1/2) * (dtdif@dtdif)**(-1/2) * np.sum(At.T@dTdif,axis=1)
        jac2 = (c2 * (mi ) ** (-1 / 2) * (dtdif @ dtdif) ** (-1 / 2) * At.T @ dTdif).T

    if (Acomb is not None) and (c3!=0):
        if (Tdif_sc.T @ Tdif_sc)==0 or c3==0:
            jac3 = 0
        else:
            jac3 = c3 * (mi ) ** (-1 / 2) * (Tdif_sc.T @ Tdif_sc) ** (-1 / 2) *  Acomb.T @ C.T  @ C @ Tdif_s
    else:
        jac3 = 0

    jac = jac + jac1 + jac2 + jac3

    return cost, jac.T.reshape(-1)
    #return jac


#========================================================================
#=============================
# Cost function for a2
def Ja2(tTr,us,vs,t,i,c1,c2):
    mi = 2**i + 1

    nt = len(t)
    if len(us.shape) == 1:
        ns = 1
    elif len(us.shape) == 2:
        ns = us.shape[1]

    tT = tTr  # .reshape((mi,ns))
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

    #Jo2 = np.sqrt(np.sum((v1 - u1) ** 2))
    #Jb2 = c1 / np.sqrt(mi) * np.sqrt(sum(Tdif ** 2)) \
    #     + c2 / np.sqrt(mi) * np.sqrt(sum(((Tdif[0:mi - 1] - Tdif[1:mi]) / (t[tc[0:mi - 1]] - t[tc[1:mi]])) ** 2))

    cost = Jo + Jb #+ Js


    return cost

#================================================================================
#================================================================================
# J_der for a3

def J_der_a3(tTr,us,vs,t,i,c1,c2,c3,Acomb=None,space_corr=None):
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

    #Tdif = (tT - T).reshape(-1)  #tT - T
    Tdif = tT - T

    # Cost
    At = np.diag(1 * np.ones(mi - 1), 1) + np.diag(-1 * np.ones(mi ), 0)
    At[mi-1,mi-1] = -1
    At /= dT
    At = csr_matrix(At)

    err = (v1 - u1).reshape(-1)
    #print(err.shape)
    Jo = np.sqrt(err@err) #np.sqrt(np.sum((v1-u1)**2))
    tdif = Tdif.reshape(-1)
    dTdif = At @ Tdif
    dtdif = dTdif.reshape(-1)
    Jb = c1 * np.sqrt(tdif@tdif/mi) \
         + c2 * np.sqrt(dtdif@dtdif/mi)

    if (Acomb is not None) and (c3!=0):
        #Tdif_s = csr_matrix(Acomb @ Tdif.T)
        Tdif_s = Acomb @ Tdif.T
        print(Tdif_s.shape)
        #print(issparse(Tdif_sc))
        print(issparse(Tdif_s))
        C = diags(np.sqrt(space_corr))
        print(C.shape)
        #C = np.diag(np.sqrt(space_corr))
        print((C @ Tdif_s).shape)
        Tdif_sc = csr_matrix((C @ Tdif_s).reshape(-1)).T
        Js = c3 * np.sqrt(Tdif_sc.T @ Tdif_sc /mi)
    else:
        Js = 0
    cost = Jo + Jb + Js

    # Derivative
    _, uT_t = mapped_weight(us,t,tT,i)
    dt = round(t[1]-t[0],2)
    dtdT = dXdT(t,i)

    #jac = - ((err@err) ** (-1 / 2)) * np.sum((((v1 - u1) * (uT_t/dt)).T @ dtdT),axis=0)

    if (err @ err)==0:
        jac = np.zeros(mi)
    else:
        jac = - ((err @ err) ** (-1 / 2)) * (((v1 - u1) * (uT_t / dt)).T @ dtdT)

    if (tdif@tdif)==0 or c1==0:
        jac1 = 0
    else:
        #jac1 = c1 * (mi*ns)**(-1/2) * (tdif@tdif)**(-1/2) * np.sum(Tdif,axis=1)
        jac1 = (c1 * (mi  ) ** (-1 / 2) * (tdif @ tdif) ** (-1 / 2) * Tdif).T


    if (dtdif@dtdif)==0 or c2==0:
        jac2 = 0
    else:
        #jac2 = c2 * (mi*ns)**(-1/2) * (dtdif@dtdif)**(-1/2) * np.sum(At.T@dTdif,axis=1)
        jac2 = (c2 * (mi ) ** (-1 / 2) * (dtdif @ dtdif) ** (-1 / 2) * At.T @ dTdif).T

    if (Acomb is not None) and (c3!=0):
        print(123)

        if (Tdif_sc.T @ Tdif_sc)==0 or c3==0:
            jac3 = 0
        else:
            print(456)
            exit()
            jac3 = c3 * (mi ) ** (-1 / 2) * (Tdif_sc.T @ Tdif_sc) ** (-1 / 2) *  Acomb.T @ C.T  @ C @ Tdif_s
            print(789)
    else:
        jac3 = 0

    jac = jac + jac1 + jac2 + jac3

    return cost#, jac.T.reshape(-1)
    #return jac


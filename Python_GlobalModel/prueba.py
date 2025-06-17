# -*- coding: utf-8 -*-
"""
Created on Fri May 30 21:38:29 2025

@author: lopez
"""

from GeneralGlobalModel_Class import GlobalModel
import Reactions_H as R
import matplotlib.pyplot as plt
import numpy as np
me=9.11e-31
mp=1.67e-27
q=1.6e-19
reactions=R.reactions1
gm2=GlobalModel(reactions,[2*mp,0],f=0.5)

def zonas(n1array,n2array,n3array):
    N_mayoritario=np.empty((n1array.shape[0],n1array.shape[1]))
    N_mayoritario[:]=np.nan
    limites=np.empty((n1array.shape[0],n1array.shape[1]))
    limites[:]=np.nan
    for i in range (n1array.shape[0]):
        for j in range (n1array.shape[1]):
            if (n1array[i,j]>n2array[i,j])&(n1array[i,j]>n3array[i,j]):
                N_mayoritario[i,j]=1
            if (n2array[i,j]>n1array[i,j])&(n2array[i,j]>n3array[i,j]):
                N_mayoritario[i,j]=2
            if (n3array[i,j]>n2array[i,j])&(n3array[i,j]>n1array[i,j]):
                N_mayoritario[i,j]=3
    for i in range(N_mayoritario.shape[0]-1):
        for j in range(N_mayoritario.shape[1]-1):
            if (N_mayoritario[i,j]!=N_mayoritario[i+1,j])|(N_mayoritario[i,j]!=N_mayoritario[i,j+1]):
                limites[i,j]=1
    return [N_mayoritario,limites]


pmin,pmax=1e5,3e6
nmin,nmax=6e13,4e14
Nn,Np=100,100
nn,pp,[ne2,nH2,n12,n22,n32,e2]=gm2.Find_solution_2D_sweep(pmin,pmax,nmin,nmax,Np,Nn)
N_mayoritario2,limites2=zonas(n12,n22,n32)

import matplotlib.colors as colors
colorsList = ['blue','red','green']
CustomCmap = colors.ListedColormap(colorsList)
bounds = np.array([0.5,1.5,2.5,3.5])
norm = colors.BoundaryNorm(boundaries=bounds, ncolors=4)

plt.figure(figsize=(8, 4))
plt.pcolormesh(nn, pp, N_mayoritario2, cmap=CustomCmap, norm=norm, shading='auto')
plt.xscale('log')
plt.yscale('log')
plt.xlim([7e13, 2.25e14])
plt.ylim([1e5, 2e6])
plt.xlabel('$n_{H_2}$ (cm$^{-3}$)')
plt.ylabel('$P_\mu$ (W/m$^{-3}$)')

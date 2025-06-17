# -*- coding: utf-8 -*-
"""
Created on Mon May  2 10:25:59 2022

@author: Mikel Elorza
"""
import numpy as np

#Constantes
me=9.11e-31
mp=1.67e-27

#Funciones para calcular las tasas de reaccion a partir de los vectores
def Heavy_t(htrr,T):
    hrr=[]
    for i in range (len(htrr)):
        s=np.double(0)
        for j in range (len(htrr[i])):
            s=s+htrr[i][j]*np.log(T)**j
        hrr.append(s)
    return hrr
def Reaction_rate(d,e):
    s=0
    for i in range (len(d)):
        s=s+d[i]*np.log(e)**i
    return np.exp(s)

#Vectores con los datos para las tasas de reaccion
a1rr=[-3.27139e1,1.35365e1,-5.7393,1.56315,-2.877e-1,3.48256e-2,-2.631976e-3,1.1195e-4,-2.039e-6]
a2rr=[-3.568e1,1.7334689e1,-7.7674,2.211579,-4.1698e-1,5.0882e-2,-3.832e-3,1.61286e-4,-2.893391e-6]
a3rr=[-1.78e1,2.278,-1.2668,4.296e-1,-9.6099e-2,1.3879e-2,-1.231e-3,6.04e-5,-1.247e-6]
a4rr=[-2.858e1,1.0385e1,-5.3838,1.95,-5.394e-1,1.007e-1,-1.161e-2,7.41162e-4,-2.001e-5]
a5rr=[-3.74619e1,1.559e1,-6.693,1.9817,-4.045e-1,5.35239e-2,-4.31745e-3,1.918e-4,-3.592e-6]
a6rr=[-3.834e1,1.426e1,-5.826,1.728,-3.598e-1,4.822e-2,-3.909e-3,1.739e-4,-3.252e-6]
a7rr=[-3.07e1,1.509e1,-7.349,2.321,-4.818e-1,6.389e-2,-5.16188e-3,2.304e-4,-4.345e-6]
a8rr=[-1.7002e1,-4.050e-1,1.019e-8,-1.6956e-8,-1.6956e-8,1.564e-10,1.979e-9,-4.395e-10,3.585e-11,-1.02e-12]
a9rr=[-1.670e1,-6.036e-1,-1.943e-8,-2.006e-7,2.963e-8,2.1342e-8,-6.3539e-9,6.152e-10,-2.03e-11]
a10rr=[[-2.095e1,-3.908e-1,-1.407e-1,-2.3302e-2],[-2.904e-1,5.612e-2,3.0341e-2,-3.656e-3],[-1.585e-1,7.312e-2,-1.768e-2,-8.362e-4],[-7.788e-2,5.829e-2,-5.71e-3,-4.779e-3]]
a11rr=[-2.81e1,1e1,-4.7719,1.467,-2.979e-1,3.8616e-2,-3.05e-3,1.33e-4,-2.4e-6]
a12rr=[-3.081e1,1.039e1,-4.2597,1.1812,-2.2775e-1,2.9e-2,-2.287e-3,1.004e-4,-1.8699e-6]
a13rr=[-3.3481997e1,1.372e1,-5.9226,1.709,-3.505e-1,4.834e-2,-4.1314e-3,1.948e-4,-3.854e-6]
a14rr=[-2.02e1,9.564e-1,-6.931e-1,1.67e-1,-3.23e-2,5.8386e-3,-8.58e-4,7.46e-5,-2.67e-6]
a15rr=[-3.454e1,1.4126e1,-6.0045,1.5895,-2.7758e-1,3.1527e-2,-2.2296e-3,8.89e-5,-1.524e-6]
a16rr=[-4.7943e1,2.629e1,-1.15e1,2.992,-4.95e-1,5.236e-2,-3.4374e-3,1.272e-4,-2.136e-6]
a17rr=[-3.408e1,1.5735e1,-6.99217,1.8522,-3.13e-1,3.3837e-2,-2.266e-3,8.565e-5,-1.39813e-6]

'''
REACCIONES:

________________Reacciones de volumen__________________________________________


Formato=['Nombre','Tipo','Reactivos[masa,carga,n]','Productos[masa,carga,n]','Perdida de energia (eV)','Reaction_rate(e)']'''

a1=['$e+H=>2e+H+$','ionization',[[me,-1,1],[mp,0,1]],[[me,-1,2],[mp,1,1]],1.36e1,lambda x:Reaction_rate(a1rr,x)]
a2=['$e+H_2=>2e+H_2^+$','ionization',[[me,-1,1],[2*mp,0,1]],[[me,-1,2],[2*mp,1,1]],1.54e1,lambda x:Reaction_rate(a2rr,x)]
a3=['$e+H_2^+=>e+H^++H$','dissociation',[[me,-1,1],[2*mp,1,1]],[[me,-1,1],[mp,1,1],[mp,0,1]],10.5,lambda x:Reaction_rate(a3rr,x)]
a4=['$e+H_2=>e+2H$','dissociation',[[me,-1,1],[2*mp,0,1]],[[me,-1,1],[mp,0,2]],1.05e1,lambda x:Reaction_rate(a4rr,x)]
a5=['$e+H_2^+=>2e+2H^+$','ionization',[[me,-1,1],[2*mp,1,1]],[[me,-1,2],[mp,1,2]],1.55e1,lambda x:Reaction_rate(a5rr,x)]
a6=['$e+H_2=>2e+H^++H$','dissociation',[[me,-1,1],[2*mp,0,1]],[[me,-1,2],[mp,0,1],[mp,1,1]],1.75e1,lambda x:Reaction_rate(a6rr,x)]
a7=['$e+H_3^+=>e+2H+H^+$','dissociation',[[me,-1,1],[3*mp,1,1]],[[me,-1,2],[mp,0,1],[mp,1,1]],1.4e1,lambda x:Reaction_rate(a7rr,x)]
a8=['$e+H_3^+=>3H$','attachment',[[me,-1,1],[3*mp,1,1]],[[mp,0,3]],0,lambda x:Reaction_rate(a8rr,x)]
a81=['$e+H_3^+=>H_2+H^*$','attachment',[[me,-1,1],[3*mp,1,1]],[[mp,0,1],[2*mp,0,1]],0,lambda x:Reaction_rate(a8rr,x)]
a9=['$e+H_2^+=>2H$','attachment',[[me,-1,1],[2*mp,1,1]],[[mp,0,2]],0,lambda x:Reaction_rate(a9rr,x)]
a100=['$H_2+H_2^+=>H_3^++H$','heavy',[[2*mp,0,1],[2*mp,1,1]],[[3*mp,1,1],[2*mp,0,1]],0,lambda x:Reaction_rate(Heavy_t(a10rr,0.2),x)]
a101=['$H_2+H_2^+=>H_3^++H$','heavy',[[2*mp,0,1],[2*mp,1,1]],[[3*mp,1,1],[2*mp,0,1]],0,lambda x:Reaction_rate(Heavy_t(a10rr,0.043),x)]
a102=['$H_2+H_2^+=>H_3^++H$','heavy',[[2*mp,0,1],[2*mp,1,1]],[[3*mp,1,1],[2*mp,0,1]],0,lambda x:Reaction_rate(Heavy_t(a10rr,0.1),x)]
a103=['$H_2+H_2^+=>H_3^++H$','heavy',[[2*mp,0,1],[2*mp,1,1]],[[3*mp,1,1],[2*mp,0,1]],0,lambda x:Reaction_rate(Heavy_t(a10rr,1),x)]
a11=['$e+H=>e+H^*$','excitation',[[me,-1,1],[mp,0,1]],[[me,-1,1],[mp,0,1]],10.2,lambda x:Reaction_rate(a11rr,x)]
a12=['$e+H_2(X^1\Sigma_g^+)=>e+H_2^*(B^1\Sigma_u^+2p\sigma)$','excitation',[[me,-1,1],[2*mp,1,1]],[[me,-1,1],[2*mp,1,1]],12.1,lambda x:Reaction_rate(a12rr,x)]
a13=['$e+H_2(X^1\Sigma_g^+)=>e+H_2^*(C^1\Pi_u^+2p\pi)$','excitation',[[me,-1,1],[2*mp,1,1]],[[me,-1,1],[2*mp,1,1]],12.4,lambda x:Reaction_rate(a13rr,x)]
a14=['$e+H_2(v=0)=>e+H_2^*(v=1)$','excitation',[[me,-1,1],[2*mp,1,1]],[[me,-1,1],[2*mp,1,1]],0.5,lambda x:Reaction_rate(a14rr,x)]
a15=['$e+H_2=>e+H+H^*$','dissociation',[[me,-1,1],[2*mp,0,1]],[[me,-1,1],[mp,0,2]],1.54e1,lambda x:Reaction_rate(a15rr,x)]
a16=['$e+H_2=>e+H^*+H^*$','dissociation',[[me,-1,1],[2*mp,0,1]],[[me,-1,1],[mp,0,2]],3.46e1,lambda x:Reaction_rate(a16rr,x)]
a17=['$e+H_2^+=>e+H^++H^*$','dissociation',[[me,-1,1],[2*mp,1,1]],[[me,-1,1],[mp,1,1],[mp,0,1]],17.5,lambda x:Reaction_rate(a17rr,x)]

'''
________________Reacciones de superficie__________________________________________


Formato=['Nombre','Tipo','Reactivos[masa,carga,n]','Productos[masa,carga,n]','Perdida de energia (eV)','gamma']'''

v1=['$H^++Pared->H$','surface',[[mp,1,1]],[[mp,0,1]],0,1]
v2=['$H_2^++Pared->H_2$','surface',[[2*mp,1,1]],[[2*mp,0,1]],0,1]
v3=['$H_3^++Pared->H+H_2$','surface',[[3*mp,1,1]],[[mp,0,1],[2*mp,0,1]],0,1]
v4=['$H+H+Pared->H_2$','surface',[[mp,0,2]],[[2*mp,0,1]],0,1]
'''
__________________________________________________________________________________________________________________

ARRAY DE LAS REACCIONES QUE SE QUIERAN ESTUDIAR'''
    
reactions1=[a1,a2,a3,a4,a5,a6,a7,a8,a81,a9,a100,a11,a12,a13,a14,a15,a16,a17,v1,v2,v3,v4]
reactions2=[a1,a2,a3,a4,a5,a6,a7,a8,a81,a9,a100,a11,a15,a16,a17,v1,v2,v3,v4]

'''Para ver la sensibilidad a la temperatura de iones'''

reactionsT0=[a1,a2,a3,a4,a5,a6,a7,a8,a81,a9,a100,a11,a12,a13,a14,a15,a16,a17,v1,v2,v3,v4]
reactionsT1=[a1,a2,a3,a4,a5,a6,a7,a8,a81,a9,a101,a11,a12,a13,a14,a15,a16,a17,v1,v2,v3,v4]
reactionsT2=[a1,a2,a3,a4,a5,a6,a7,a8,a81,a9,a102,a11,a12,a13,a14,a15,a16,a17,v1,v2,v3,v4]
reactionsT3=[a1,a2,a3,a4,a5,a6,a7,a8,a81,a9,a103,a11,a12,a13,a14,a15,a16,a17,v1,v2,v3,v4]


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
a1rr=[-4.4098e1,2.3915e1,-1.075e1,3.058e0,-5.68e-1,6.795e-2,-5.01e-3,2.067e-4,-3.649e-6]
a2rr=[-6.87e1,4.3933e1,-1.848e1,4.7e0,-7.692e-1,8.11e-2,-5.324e-3,1.975e-4,-2.165e-6]
a3rr=[-4.076450793433e1,1.847216050626e+1,-7.553534847500,1.936036716566,-3.278509524847e-1,3.627243238755e-02,-2.509995616613e-03,9.829302307697e-05,-1.659516418994e-06]
a4rr=[-4.439802014466e+1,2.170928173941e1,-9.582182742745,2.631183876455,-4.687644097236e-1,5.361959275824e-02,-3.786601021417e-03,1.500190728471e-04,-2.548241584846e-06]
a5rr=[-4.439802014466e1,2.170928173941e1,-9.582182742745,2.631183876455,-4.687644097236e-1,5.361959275824e-02,-3.786601021417e-03,1.500190728471e-04,-2.548241584846e-06 ]
a6rr=[-3.944902284550e1,1.801440475215e1,-7.941287139217,2.108879073816,-3.656365579422e-1,4.13041820959e-02,-2.921666597766e-03,1.171506777488e-04,2.027094391374e-06 ]
'''
REACCIONES:

________________Reacciones de volumen__________________________________________


Formato=['Nombre','Tipo','Reactivos[masa,carga,n]','Productos[masa,carga,n]','Perdida de energia (eV)','Reaction_rate(e)']'''

a1=['e+He=>2e+He+','ionization',[[me,-1,1],[4*mp,0,1]],[[me,-1,2],[4*mp,1,1]],3.16,lambda x:Reaction_rate(a1rr,x)]
a2=['e+He+=>2e+He2+','ionization',[[me,-1,1],[4*mp,1,1]],[[me,-1,2],[4*mp,2,1]],7.94,lambda x:Reaction_rate(a2rr,x)]
a3=['e+He=>e+He+*','excitation',[[me,-1,1],[4*mp,0,1]],[[me,-1,1],[4*mp,0,1]],3.98,lambda x:Reaction_rate(a3rr,x)]
a4=['e+He=>e+He+*','excitation',[[me,-1,1],[4*mp,0,1]],[[me,-1,1],[4*mp,0,1]],5.01,lambda x:Reaction_rate(a4rr,x)]
a5=['e+He=>e+He+*','excitation',[[me,-1,1],[4*mp,0,1]],[[me,-1,1],[4*mp,0,1]],5.01,lambda x:Reaction_rate(a5rr,x)]
a6=['e+He=>e+He+*','excitation',[[me,-1,1],[4*mp,0,1]],[[me,-1,1],[4*mp,0,1]],5.01,lambda x:Reaction_rate(a6rr,x)]


'''
________________Reacciones de superficie__________________________________________


Formato=['Nombre','Tipo','Reactivos[masa,carga,n]','Productos[masa,carga,n]','Perdida de energia (eV)','gamma']'''

v1=['He++wall->He','surface',[[4*mp,1,1]],[[4*mp,0,1]],0,1]
v2=['He2++wall->He','surface',[[4*mp,2,1]],[[4*mp,0,1]],0,1]




'''ARRAY DE LAS REACCIONES QUE SE QUIERAN ESTUDIAR'''
    
reactions=[a1,a2,a3,a4,a5,a6,v1,v2]

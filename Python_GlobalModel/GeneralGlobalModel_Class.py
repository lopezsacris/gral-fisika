# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:39:36 2022

@author: Mikel Elorza
"""

import numpy as np
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')
q=1.6e-19
me=9.11e-31
mp=1.67e-27

'''
This class is designed to find the solution of a global model for low density plasma.
Attributes:
    -reactions
    -variable_specie
    -volume_reactions
    -surface_reactions
    -species
    -ions
    -neutrals
    -L
    -R
    -V_chamber
    -A_chamber
    -V_plasma
    -A_plasma
    -f
    -gamma
    -T_i
    -T_g
Methods:
    -Find_solution
    -Find_solution_p_sweep
    -Find_solution_n_sweep
    -Find_solution_2D_sweep
    -Contributions
    -Contributions_n_sweep
    -Contributions_p_sweep
    -Contributions_2D_sweep
    -Absortion
    -Absortion_n_sweep
    -Absortion_p_sweep
    -Absortion_2D_sweep
'''

class GlobalModel:
    '''
    This method initializes the attributes of the object.
    Args:
        - Reactions: list of the reactions. There are volume and surface reactions.
            Format of the volume reactions: ['name','type of the reaction',[reactants:[mass, charge,#]],[products:[mass,charge,#]],energy gain, Function of reaction rate (e) in cm^3/s]
            Format of the volume reactions: ['name','surface',[reactants:[mass, charge,#]],[products:[mass,charge in units of q,#]],0, 0]
        - variable_specie: the specie whose density is an input.
        - L: lenght of the plasma chamber in cm
        - R: Radius of the plasma chamber in cm
        - gamma: recombination coeff for atomic hydrogen in the material of the plasma chamber.
    '''
    def __init__(self,reactions,variable_specie,L=10,R=3.1,gamma=0.1,Ti=0.2,Tg=500,f=1):
        self.reactions=reactions
        self.Variable_specie=variable_specie
        self.Volume_reactions,self.surface_reactions=self.__Clasify_reactions()
        self.species=self.__Species_list()
        self.ions=self.__Ions()
        self.neutrals=self.__Neutrals()
        self.L=L
        self.R=R
        self.f=f
        self.V_chamber=np.pi*R**2*L
        self.A_chamber=2*np.pi*R*L+2*np.pi*R**2
        self.V_plasma=self.f*self.V_chamber
        self.A_plasma=self.f**(2/3)*self.A_chamber
        self.gamma=gamma
        self.Ti=Ti
        self.Tg=Tg
    
    '''
    This method separates surface and volume reactions.     
    '''
    def __Clasify_reactions(self):
        volume_reactions=[]
        surface_reactions=[]
        for reaction in self.reactions:
            if reaction[1]=='surface':
                surface_reactions.append(reaction)
            else:
                volume_reactions.append(reaction)
        return [volume_reactions,surface_reactions]
    
    '''
    This method reads all the reactions and return a list of all the different species.
    The list is ordered by mass (if same mass by charge), so the element [0] is the electron.   
    '''
    def __Species_list(self):
        species=[]
        for reaction in self.reactions:
            for elem in reaction[2]:
                spc=elem[0:2]
                if spc not in species:
                    species.append(spc)
            for elem in reaction[3]:
                spc=elem[0:2]
                if spc not in species:
                    spc=elem[0:2]
                    species.append(spc)
        species.sort()
        return species
        
    '''
    This method reads all the species in species and return a list of all the different ions.  
    '''
    def __Ions(self):
        ions=[]
        for specie in self.species:
            if specie[1]>0:
                ions.append(specie)
        return ions
    
    '''
    This method reads all the species in species_list and return a list of all the different neutrals.  
    '''
    def __Neutrals(self):
        neutrals=[]
        for specie in self.species:
            if specie[1]==0:
                neutrals.append(specie)
        return neutrals 
        
    '''
    Private method. Bohm velocity of the ions.
    Args:
        -Te: Electron temperature in eV
        -M: Ion mass in kg
    Returns:
        - Bohm velocity in cm/s.
    '''
    def __Bohm(self,Te,M):
        return np.sqrt(Te/M*q*10000)
    
    '''
    Private method. Recombination rate of the atomic hydrogen.
    Args:
        -e: Electron temperature in eV
        -n0: Neutral gas density in cm^-3
    Returns:
        - Recombination rate of atomic hydrogen in 1/s.
    '''  
    def __Recombination_rate(self,e,n0):
        Dh=1.28e20/n0*np.sqrt(self.Tg/300)
        A0_2=1/(2.405/self.R)**2+(np.pi/self.L)**2
        vth=np.sqrt(8/np.pi*self.Tg*1.38e-23*10000/mp*3/2) #*3/2 viene de la masa relativa entre hidrogeno atomico y molecular
        tau_rec=2*(2-self.gamma)*self.V_chamber/self.A_chamber/self.gamma/vth+A0_2/Dh
        return 1/tau_rec
    '''
    Private method. Calculates the effective surface of the plasma, and returns its ratio to the volume.
    Args:
        -n0: Neutral gas density in cm^-3
    Returns:
        - Effective surface / Plasma chamber volume (1/cm)
    '''  
    def __Aeff_V(self,n0):
        mfp=1/n0/5e-15
        hl=0.86/np.sqrt(3+self.L/2/mfp)
        hr=0.8/np.sqrt(4+self.R/mfp)
        Aeff=2*np.pi*self.R**2*hl+2*np.pi*self.R*self.L*hr
        Aeff=2*np.pi*self.R**2*hl
        self.aeff_V=Aeff/self.V_chamber
        return Aeff/self.V_chamber
    '''
    Private method. Given a list of the concentration of the species, calculates the total charge. When the solution is found Neutrality=0
    Args:
        -concentrations: list of densities of the species in cm^-3. Species ordered by mass (if equal mass, by charge).
    Returns:
        - Total charge in units of q.
    ''' 
    def __Neutrality(self,concentrations):
        s=0
        for i in range(len(self.species)):
            s+=self.species[i][1]*concentrations[i]
        return s
    '''
    Private method. Calculates the power not absorbed in in the plasma. When the solution found is Power_absortion=0
    Args:
        -concentrations: list of densities of the species in cm^-3. Species ordered by mass (if equal mass, by charge).
        -pwr: Power transmited to the plasma chamber (W/m^3).
        -e: Electron temperature (eV).
    Returns:
        - Power not absorbed in the plasma chamber in w/m^3.
    '''
    def __Power_absortion(self,conc,pwr,e):
        absorbed=0
        sum_nub=0
        for reaction in self.Volume_reactions:
            if reaction[1]=='excitation':
                reac=reaction[2]
                reac.sort() 
                specie=reac[1][0:2]#The specie that its not the electron
                i=self.species.index(specie)
                absorbed+=conc[0]*conc[i]*reaction[5](e)*reaction[4]
            if reaction[1]=='ionization':
                reac=reaction[2]
                reac.sort() 
                specie=reac[1][0:2]#The specie that its not the electron
                i=self.species.index(specie)
                if e<3/2*reaction[4]:
                    deltae=1/2*(e-reaction[4])
                else:
                    deltae=1/4*reaction[4]
                absorbed+=conc[0]*conc[i]*reaction[5](e)*(reaction[4]+deltae)
            if reaction[1]=='attachment':
                reac=reaction[2]
                reac.sort() 
                specie=reac[1][0:2]
                i=self.species.index(specie)
                absorbed+=conc[0]*conc[i]*reaction[5](e)*e
            if reaction[1]=='elastic':
                reac=reaction[2]
                reac.sort() 
                specie=reac[1][0:2]
                i=self.species.index(specie)
                absorbed+=conc[0]*conc[i]*reaction[5](e)*3*me/specie[0]*e
            if reaction[1]=='dissociation':
                reac=reaction[2]
                reac.sort()
                specie1=reac[0][0:2]
                specie2=reac[1][0:2]
                i=self.species.index(specie1)
                j=self.species.index(specie2)
                absorbed+=conc[0]*conc[j]*reaction[5](e)*3*me/specie[0]*e
        for ion in self.ions:
            i=self.species.index(ion)
            sum_nub+=conc[i]*self.__Bohm(e,ion[0])
        Vs=-e*np.log(4*sum_nub/conc[0]*np.sqrt(np.pi*me/8/(e*q*10000)))
        absorbed+=(5/2*e+Vs)*sum_nub*self.f**(-1/3)*self.__Aeff_V(conc[3])
        absorbed=absorbed*q*1e6 #To put in W/m^3.
        return pwr-absorbed
    
    '''
    Private method. Calculates the rate of the particle generation for each specie.  Whe the solution is found Specie_balance=[0,0...,0].
    Args:
        -concentrations: list of densities of the species in cm^-3. Species ordered by mass (if equal mass, by charge).
        -e: Electron temperature (eV).
    Returns:
        -List of the rate of particle generation (1/s/cm^3). Particles are ordered by mass (when equal mass by charge).
    '''
    def __Specie_balance(self,concentrations,e):
        equations=np.zeros(len(self.species))
        n0=0
        for neu in self.neutrals:
            i=self.species.index(neu)
            n0+=concentrations[i]
        #Volume reactions
        V=np.zeros((len(self.species),len(self.species),len(self.species)))
        S=np.zeros((len(self.species),len(self.species)))
        for reaction in self.Volume_reactions:
            reactant_energy=e
            if reaction[1]=='heavy':
                reactant_energy=self.Ti
            reac1=reaction[2][0][0:2]
            reac2=reaction[2][1][0:2]
            i=self.species.index(reac1)
            j=self.species.index(reac2)
            if i>j:
                i,j=j,i
                reac1,reac2=reac2,reac1
            V[i,j,i]-=concentrations[i]*concentrations[j]*reaction[5](reactant_energy)#losses
            V[i,j,j]-=concentrations[i]*concentrations[j]*reaction[5](reactant_energy)
            for l in range(len(reaction[3])):
                prod=reaction[3][l][0:2]
                n=reaction[3][l][2]
                k=self.species.index(prod)
                V[i,j,k]+=n*concentrations[i]*concentrations[j]*reaction[5](reactant_energy)#gains
        for k in range(len(self.species)):
            equations[k]+=sum(sum(V[:,:,k])) # Sum all the reactions involving  specie[k]
        #Surface reactions
        for reaction in self.surface_reactions:
                reac=reaction[2][0][0:2]
                nr=reaction[2][0][2]
                i=self.species.index(reac)
                if reac in self.ions:
                    S[i,i]-=nr*concentrations[i]*self.__Bohm(e,reac[0])*self.f**(-1/3)*self.__Aeff_V(n0)
                if reac in self.neutrals:
                    S[i,i]-=nr*concentrations[i]*self.__Recombination_rate(e,n0)
                for l in range(len(reaction[3])):
                    prod=reaction[3][l][0:2]
                    n=reaction[3][l][2]
                    k=self.species.index(prod)
                    if reac in self.ions:
                        S[i,k]+=n*concentrations[i]*self.__Bohm(e,reac[0])*self.f**(-1/3)*self.__Aeff_V(n0)
                    if reac in self.neutrals:
                        S[i,k]=n*concentrations[i]*self.__Recombination_rate(e,n0)
        for k in range(len(self.species)):
            equations[k]+=sum(S[:,k])
        #Delete the equations of the electron and the variable specie.
        var_index=self.species.index(self.Variable_specie)
        equations=np.delete(equations,var_index)
        equations=np.delete(equations,0)
        return equations
    '''
    Private method. Calculates the equation system of the problem for a given solution, pwr and variable_species's gas density. The solution is found when Eq_sys=0.
    Args:
        -solution: [densities of the species, Te]. In the case of hydrogen[ne,nH,nH+,nH2,nH2+,nH3+,Te]
        -pwr: Power transmited to the plasma chamber (W/m^3).
        -e: Electron temperature (eV).
    Returns:
        - List with the results of Neutrality(), Power_absortion() and Specie_balance().
    '''
    def Eq_system(self,solution,pwr,n0):
        e=solution[-1]
        var_index=self.species.index(self.Variable_specie)
        concentrations=solution[:-1]
        concentrations=np.insert(concentrations,var_index,n0)
        neutrality_eq=[self.__Neutrality(concentrations)]
        pwr_eq=[self.__Power_absortion(concentrations,pwr,e)]
        species_eq=self.__Specie_balance(concentrations,e)
        sys=np.append(species_eq,(neutrality_eq,pwr_eq))
        return sys
    '''
    Proposes a new initial solution starting from the previous initial solution. This new solution is the old one with the energy increased 0.5eV
    Args:
        -init: previous initial solution
    Returns:
        -new_init: previous initial solution
    '''
    '''def New_init(self,init):
        Deltae=0.5
        new_init=init
        new_init[-1]=init[-1]+Deltae
        return new_init'''
       
    '''
    Find the solution of the problem: the root of Eq_sys(). It starts with the initial solution init, and it varies its energy until it finds the root.
    Args:
        -pwr: Power transmited to the plasma chamber (W/m^3).
        -n: Density of H2 in cm^-3.
    Returns:
        - The solution of the problem. sol=[ne,nH,nH+,nH2,nH2+,nH3+,Te]
    '''
    def Find_solution(self,pwr,n): #cambia la init. La especie neutra en mayor concentración. O que la init se pueda cambiar desde fuera o que sea un input del objeto
        e0=0.8
        init=n/100*np.ones(len(self.species))
        init[-1]=e0
        sol=fsolve(self.Eq_system,init,args=(pwr,n),full_output=1)
        converge=sol[2]
        while (converge!=1)&(e0<120):
            e0+=0.5
            init[-1]=e0
            sol=fsolve(self.Eq_system,init,args=(pwr,n),full_output=1)
            converge=sol[2]
        return sol
    '''
    Solution of the problem mantaining pwr constant and varying density of the variable_specie.
    Args:
        -p0: Power transmited to the plasma chamber (W/m^3).
        -n0: minimum density of the variable specie of the sweep  (cm^-3).
        -n1: maximum density of the variable specie of the sweep  (cm^-3).
        -N: number of points.
    Returns:
        - ns: array of the densities of variable specie
        - sol_matrix: array with solutions of the sweep. in case of Hydrogen [array of ne, array of nH, array of nH+...,array of Te]
    '''
    def Find_solution_n_sweep(self,p0,n0,n1,N,scale='linear'):
        if scale=='linear':
            ns=np.linspace(n0,n1,N)
        if scale=='log':
            ns=np.logspace(np.log10(n0),np.log10(n1),N)
        length=self.Find_solution(p0,n1)[0].size
        sol_matrix=np.zeros((length,N))
        for i in range (N):
            ni=ns[i]
            sol=self.Find_solution(p0,ni)
            for j in range (length):
                sol_matrix[j,i]=sol[0][j]
        return [ns,sol_matrix]
    '''
    Solution of the problem mantaining the density of the variable_specie constant and varying power transmited to the plasma chamber.
    Args:
        -n0: density of the variable_specie (cm^-3).
        -p0: minimun power of the sweep  (W/m^3).
        -p1: maximum power of the sweep  (W/m^3).
        -N: number of points.
    Returns:
        - ps: array of powers
        - sol_matrix: array with solutions of the sweep. in case of Hydrogen [array of ne, array of nH, array of nH+...,array of Te]
    '''
    def Find_solution_p_sweep(self,n0,p0,p1,N,scale='linear'):
        if scale=='linear':
            ps=np.linspace(p0,p1,N)
        if scale=='log':
            ps=np.logspace(np.log10(p0),np.log10(p1),N)
        length=self.Find_solution(p0,n0)[0].size
        sol_matrix=np.zeros((length,N))
        for i in range (N):
            pi=ps[i]
            sol=self.Find_solution(pi,n0)
            for j in range (length):
                sol_matrix[j,i]=sol[0][j]
        return [ps,sol_matrix]
    '''
    Solution of the problem varying both H2 density and power.
    Args:
        -n0i: minimum density of the variable_specie of the sweep(cm^-3).
        -n0f: maximum density of the variable_specie of the sweep(cm^-3).
        -p0i: minimum power of the sweep  (W/m^3).
        -p0f: maximum power of the sweep  (W/m^3).
        -xN: number of points in power sweep.
        -yN: number of points in H2 density sweep.
    Returns:
        - nn: xN x yN mesh of densities of the variable_specie
        - pp: xN x yN mesh of power 
        - sol_matrix: array with solutions of the sweep. in case of Hydrogen [array of ne, array of nH, array of nH+...,array of Te]
    '''
    def Find_solution_2D_sweep(self,p0i,p0f,n0i,n0f,xN,yN,scale='linear'):
        if scale=='linear':
            p0=np.linspace(p0i,p0f,yN)
            n0=np.linspace(n0i,n0f,xN)
        if scale=='log':
            p0=np.logspace(np.log10(p0i),np.log10(p0f),yN)
            n0=np.logspace(np.log10(n0i),np.log10(n0f),xN)
        pp,nn=np.meshgrid(p0,n0)
        length=self.Find_solution(p0i,n0i)[0].size
        sol_matrix=np.zeros((length,xN,yN))
        for j in range (yN):
            for i in range(n0.size):
                sol=self.Find_solution(p0[j],n0[i])
                for k in range (length):
                    sol_matrix[k,i,j]=sol[0][k]
        return [nn,pp,sol_matrix]
    '''
    Calculates the rate of particle generation of specie 'specie' with each of the reactions for the solution Find_solution(pwr,n0).
    Args:
        -specie: the specie that you are looking for ([mass,charge]).
        -pwr: The power transmited to the chamber (W/cm^3).
        -n0: H2 density (cm^-3).
    Returns:
        - List of the names of the reactions.
        - The rate of particle generation of 'specie' with each reaction (1/s/cm^3). 
    '''
    def Contributions(self,specie,pwr,nH2):#solution:[ne,nH,nH+,nH2+,nH3+,e]
        sol0=self.Find_solution(pwr,nH2)[0]
        e=sol0[-1]
        var_index=self.species.index(self.Variable_specie)
        concentrations=sol0[:-1]
        concentrations=np.insert(concentrations,var_index,nH2)
        names=[]
        values=[]
        n0=0
        for neu in self.neutrals:
            i=self.species.index(neu)
            n0+=concentrations[i]
        for reaction in self.Volume_reactions:
            reactant_energy=e
            if reaction[1]=='heavy':
                reactant_energy=self.Ti
            reac1=reaction[2][0][0:2]
            reac2=reaction[2][1][0:2]
            i=self.species.index(reac1)
            j=self.species.index(reac2)
            value=0
            for spc in reaction[3]:
                prod=spc[0:2]
                n=spc[2]
                if specie==prod:
                    value=+n*concentrations[i]*concentrations[j]*reaction[5](reactant_energy)
            for spc in reaction[2]:
                reac=spc[0:2]
                n=spc[2]
                if specie==reac:
                    value-=n*concentrations[i]*concentrations[j]*reaction[5](reactant_energy)
            if value!=0:
                names.append(reaction[0])
                values.append(value)
        for reaction in self.surface_reactions:
            reac=reaction[2][0][0:2]
            nr=reaction[2][0][2]
            i=self.species.index(reac)
            value=0
            if reac==specie:
                if reac in self.ions:
                    value-=nr*concentrations[i]*self.__Bohm(e,reac[0])*self.__Aeff_V(n0)
                if reac in self.neutrals:
                    value-=nr*concentrations[i]*self.__Recombination_rate(e,n0)
            for l in range(len(reaction[3])):
                prod=reaction[3][l][0:2]
                n=reaction[3][l][2]
                if prod==specie:
                    if reac in self.ions:
                        value+=n*concentrations[i]*self.__Bohm(e,reac[0])*self.__Aeff_V(n0)
                    if reac in self.neutrals:
                        value+=n*concentrations[i]*self.__Recombination_rate(e,n0)
            if value!=0:
                names.append(reaction[0])
                values.append(value)
        return [names,values]
        
        
    '''
    Calculates the contributions of each reaction in a power sweep.
    Args:
        -specie: the specie that you are looking for ([mass,charge]).
        -n0: density of the variable_specie (cm^-3).
        -p0: minimum power in the sweep (W/cm^3).
        -pf: maximum power in the sweep (W/cm^3).
        -N: number of points.
        
    Returns:
        - Array of powers in the sweep.
        - List of the names of the reactions.
        - The array of the rates of particle generation of 'specie' with each reaction (1/s/cm^3). 
    '''
    def Contributions_p_sweep(self,specie,n0,p0,pf,N,scale='linear'):
        names0,value0=self.Contributions(specie,n0,p0)
        values0=np.zeros((N,len(names0)))
        if scale=='linear':
            ps=np.linspace(p0,pf,N)
        if scale=='log':
            ps=np.logspace(np.log10(p0),np.log10(pf),N)
        for i in range(N):
            value0=self.Contributions(specie,ps[i],n0)[1]
            values0[i]=value0
        return [ps,names0,values0]
        
    '''
    Calculates the contributions of each reaction in a sweep of the density of the variable_specie.
    Args:
        -specie: the specie that you are looking for ([mass,charge]).
        -p0: The power transmited to the chamber (W/cm^3).
        -n0: minimum density of the variable_specie in the sweep (cm^-3).
        -nf: maximum density of the variable_specie in the sweep (cm^-3).
        -N: number of points.
        
    Returns:
        - Array of density of the variable_specie in the sweep.
        - List of the names of the reactions.
        - The array of the rates of particle generation of 'specie' with each reaction (1/s/cm^3). 
    '''
    def Contributions_n_sweep(self,specie,p0,n0,nf,N,scale='linear'):
        names0,value0=self.Contributions(specie,n0,p0)
        values0=np.zeros((N,len(names0)))
        if scale=='linear':
            ns=np.linspace(n0,nf,N)
        if scale=='log':
            ns=np.logspace(np.log10(n0),np.log10(nf),N)
        for i in range(N):
            value0=self.Contributions(specie,p0,ns[i])[1]
            values0[i]=value0
        return [ns,names0,values0]
    '''
    Calculates the contributions of each reaction in a sweep of density of the variable_specie and power.
    Args:
        -specie: the specie that you are looking for ([mass,charge]).
        -p0: minimum power in the sweep (W/cm^3).
        -pf: maximum power in the sweep (W/cm^3).
        -n0: minimum density of the variable_specie in the sweep (cm^-3).
        -nf: maximum density of the variable_specie in the sweep (cm^-3).
        -Np: number of points in the power sweep.
        -Nn: number of points in the density sweep.
        
    Returns:
        - Np x Nn np array of density of the variable_specie in the sweep.
        - Np x Nn np array of powers in the sweep.
        - List of the names of the reactions.
        - Np x Nn x len(names): The array of the rates of particle generation of 'specie' with each reaction (1/s/cm^3). 
    '''
    def Contributions_2D_sweep(self,specie,p0,pf,n0,nf,Np,Nn,scale='linear'):
        if scale=='linear':
            ps=np.linspace(p0,pf,Np)
            ns=np.linspace(n0,nf,Nn)
        if scale=='log':
            ps=np.logspace(np.log10(p0),np.log10(pf),Np)
            ns=np.logspace(np.log10(n0),np.log10(nf),Nn)
        nn,pp=np.meshgrid(ns,ps)
        names0,value0=self.Contributions(specie,p0,n0)
        values0=np.zeros((Np,Nn,len(names0)))
        for i in range(Np):
            for j in range(Nn):
                value0=self.Contributions(specie,ps[i],ns[j])[1]
                values0[i,j]=value0
        return [nn,pp,names0,values0]
    '''
    This method calculates the power absorbed by each reaction in w/m^3.
    Args:
        -pwr: The power transmited to the chamber (W/cm^3).
        -n0: density of the variable_specie (cm^-3).
    Returns:
        - List of the names of the reactions.
        - The energy absorbed with each reaction (W/m^3). 
    '''
    def Absortion(self,pwr,n):
        sol0=self.Find_solution(pwr,n)[0]
        e=sol0[-1]
        var_index=self.species.index(self.Variable_specie)
        conc=sol0[:-1]
        conc=np.insert(conc,var_index,n)
        absorbed=0
        names=[]
        values=[]
        for reaction in self.Volume_reactions:
            if reaction[1]=='excitation':
                reac=reaction[2]
                reac.sort() 
                specie=reac[1][0:2]#quiero la especie que no es el electron
                i=self.species.index(specie)
                names.append(reaction[0])
                values.append(conc[0]*conc[i]*reaction[5](e)*reaction[4]*q*1e6)
                absorbed+=conc[0]*conc[i]*reaction[5](e)*reaction[4]
            if reaction[1]=='ionization':
                reac=reaction[2]
                reac.sort()
                specie=reac[1][0:2]
                i=self.species.index(specie)
                if e<3/2*reaction[4]:
                    deltae=1/2*(e-reaction[4])
                else:
                    deltae=1/4*reaction[4]
                absorbed+=conc[0]*conc[i]*reaction[5](e)*(reaction[4]+deltae)
                names.append(reaction[0])
                values.append(conc[0]*conc[i]*reaction[5](e)*(reaction[4]+deltae)*q*1e6)
            if reaction[1]=='attachment':
                reac=reaction[2]
                reac.sort() 
                specie=reac[1][0:2]
                i=self.species.index(specie)
                absorbed+=conc[0]*conc[i]*reaction[5](e)*e
                names.append(reaction[0])
                values.append(conc[0]*conc[i]*reaction[5](e)*e*q*1e6)
            if reaction[1]=='elastic':
                reac=reaction[2]
                reac.sort() 
                specie=reac[1][0:2]
                i=self.species.index(specie)
                absorbed+=conc[0]*conc[i]*reaction[5](e)*3*me/specie[0]*e
                names.append(reaction[0])
                values.append(conc[0]*conc[i]*reaction[5](e)*3*me/specie[0]*e*q*1e6)
            if reaction[1]=='dissociation':
                reac=reaction[2]
                reac.sort()
                specie1=reac[0][0:2]
                specie2=reac[1][0:2]
                i=self.species.index(specie1)
                j=self.species.index(specie2)
                absorbed+=conc[i]*conc[j]*reaction[5](e)*reaction[4]
                names.append(reaction[0])
                values.append(conc[i]*conc[j]*reaction[5](e)*reaction[4]*q*1e6)
        return [names,values]
    '''
    Calculates the power absortion of each reaction in sweep of the density of the variable_specie.
    Args:
        -p0: The power transmited to the chamber (W/cm^3).
        -n0: minimum density of the variable_specie in the sweep (cm^-3).
        -nf: maximum density of the variable_specie in the sweep (cm^-3).
        -N: number of points.
        
    Returns:
        - Array of densities of the variable_specie in the sweep.
        - List of the names of the reactions.
        - The array of the energy absorbed with each reaction for each density (W/m^3).  
    '''
    def Absortion_n_sweep(self,p0,n0,nf,N,scale='linear'):
        names0,value0=self.Absortion(p0,n0)
        values0=np.zeros((N,len(names0)))
        if scale=='linear':
            ns=np.linspace(n0,nf,N)
        if scale=='log':
            ns=np.logspace(np.log10(n0),np.log10(nf),N)
        for i in range(N):
            value0=self.Absortion(p0,ns[i])[1]
            values0[i]=value0
        return [ns,names0,values0]
    '''
    Calculates the power absortion of each reaction in a power sweep.
    Args:
        -n0: density of the variable_specie (cm^-3).
        -p0: minimum power in the sweep (W/cm^3).
        -pf: maximum power in the sweep (W/cm^3).
        -N: number of points.
        
    Returns:
        - Array of powers in the sweep.
        - List of the names of the reactions.
        - The array of the energy absorbed with each reaction for each density (W/m^3). 
    '''
    def Absortion_p_sweep(self,n0,p0,pf,N,scale='linear'):
        names0,value0=self.Absortion(p0,n0)
        values0=np.zeros((N,len(names0)))
        if scale=='linear':
            ps=np.linspace(p0,pf,N)
        if scale=='log':
            ps=np.logspace(np.log10(p0),np.log10(pf),N)
        for i in range(N):
            value0=self.Absortion(ps[i],n0)[1]
            values0[i]=value0
        return [ps,names0,values0]
    '''
    Calculates the power absortion of each reaction in a sweep of H2 density and power.
    Args:
        -p0: minimum power in the sweep (W/cm^3).
        -pf: maximum power in the sweep (W/cm^3).
        -n0: minimum density of the variable_specie in the sweep (cm^-3).
        -nf: maximum density of the variable_specie in the sweep (cm^-3).
        -Np: number of points in the power sweep.
        -Nn: number of points in the density sweep.
        
    Returns:
        - Np x Nn np array of densities of the variable_specie in the sweep.
        - Np x Nn np array of powers in the sweep.
        - List of the names of the reactions.
        - Np x Nn x len(names): The array of the power absortions of each reaction (W/m^3). 
    '''
    def Absortion_2D_sweep(self,p0,pf,n0,nf,Np,Nn,scale='linear'):
        if scale=='linear':
            ps=np.linspace(p0,pf,Np)
            ns=np.linspace(n0,nf,Nn)
        if scale=='log':
            ps=np.logspace(np.log10(p0),np.log10(pf),Np)
            ns=np.logspace(np.log10(n0),np.log10(nf),Nn)
        nn,pp=np.meshgrid(ns,ps)
        names0,value0=self.Absortion(p0,n0)
        values0=np.zeros((Np,Nn,len(names0)))
        for i in range(Np):
            for j in range(Nn):
                value0=self.Absortion(ps[i],ns[j])[1]
                values0[i,j]=value0
        return [nn,pp,names0,values0]
    
    
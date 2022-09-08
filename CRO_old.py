# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:20:09 2022

@author: LuisMi-ISDEFE
"""
import numpy as np
import matplotlib.pyplot as plt
import time

class CROClass:
    def __init__(self,
                Ngen=500,
                M=20,
                N=15,
                L=1000,
                rho=0.6,
                Fb=0.9,
                Fa=0.01,
                k=3,
                Pd=0.1,
                K=20,
                opt='max'):
        self.Ngen = Ngen
        self.M = M
        self.N = NcroSL
        self.L = L
        self.rho = rho
        self.Fb = Fb
        self.Fa = Fa
        self.Fd = Fa
        self.k = k
        self.Pd = Pd
        self.K = K
        self.opt = opt
    
    def ImprimirInformacion(self):
        print("Los parámetros del CRO son los siguientes:")
        print("Número de generaciones: " + str(self.Ngen))
        print("Tamaño del coral: " + str(self.M) + "x" + str(self.N))
        print("Tamaño de los individuos: " + str(self.L))
        print("Proporción de ocupación inicial: " + str(self.rho))
        print("Proporción de broadcast spawning: " + str(self.Fb))
        print("Proporción de reproducción asexual: " + str(self.Fa))
        print("Intentos máximos para asentarse: " + str(self.k))
        print("Proporción de corales depredados: " + str(self.Pd))
        print("Corales iguales permitidos como máximo: " + str(self.K))
        print("Modo de optimización: " + self.opt)
        
    def ReefInitialization(self, M = None, N = None, rho = None, L = None):
        if M == None:
            M = self.M
        if N == None:
            N = self.N
        if rho == None:
            rho = self.rho
        if L == None:
            L = self.L
          
        REEF = np.zeros([M,N])
        REEFpob = np.zeros([M*N,L])
    
        O = round(M*N*rho)
        a = np.random.permutation(M*N)
        ocup = a[0:O]
        fil = (ocup%M).astype(int)
        col = (ocup/M).astype(int)
        
        REEF[fil,col] = 1
        REEFpob[ocup,:] = 1 * (np.random.rand(O,L) > 0.5) # Estos son los individuos, hay que modificar esta línea cuando trabajemos con otra codificación
        REEF = REEF.astype(int)
        REEFpob = REEFpob.astype(int)
        return REEF, REEFpob
    
    def ReefFitness(self, REEFpob):
        REEFcost = sum(REEFpob.transpose())
        return REEFcost
        
    def BroadcastSpawning(self, REEF, REEFpob, Fb = None, tipo = "bin"):
        if Fb == None:
            Fb = self.Fb
            
        nspawners = int(round(Fb*sum(sum(REEF))))
        if nspawners % 2 != 0:
            nspawners -= 1
            
        p = np.where(REEF != 0)
        p = self.M*p[1]+p[0]
        a = np.random.permutation(len(p))
        
        aux = int(nspawners/2)
        spawners = p[a[0:nspawners]]
        spawners1 = REEFpob[spawners[0:int(aux)],:]
        spawners2 = REEFpob[spawners[int(aux):nspawners],:]
        
        mask = np.random.randint(0,2,size=np.shape(spawners1),dtype=int)
        
        ESlarvae1 = spawners1*(1-mask)+spawners2*mask
        ESlarvae2 = spawners2*(1-mask)+spawners1*mask
        ESlarvae = np.concatenate((ESlarvae1,ESlarvae2),axis=0)
        return ESlarvae
            
    def Brooding(self, REEF, REEFpob, Fb = None, tipo = "bin"):
        if Fb == None:
            Fb = self.Fb
        
        nbrooders = int(round((1-Fb)*sum(sum(REEF))))
        
        p = np.where(REEF != 0)
        p = self.M*p[1]+p[0]
        a = np.random.permutation(len(p))

        brooders = p[a[0:nbrooders]]
        brooders = REEFpob[brooders[0:nbrooders],:]
        
        aux = np.random.randint(0,2, size=np.shape(brooders))
        
        ISlarvae = (brooders + aux) % 2
        ISlarvae = ISlarvae.astype(int)
        return ISlarvae
                
    def LarvaeCorrection(self,larvae):
        # print("Larvae Correction")
        return larvae

    def LarvaeSetting(self, REEF, REEFpob, REEFcost, larvae, larvaecost, k0 = None, opt = None):
        if k0 == None:
            k0 = self.k
        if opt == None:
            opt = self.opt

        Nlarvae = np.shape(larvae)[0]
        P = np.shape(REEFpob)[0]
        a = np.random.permutation(Nlarvae)
        larvae = larvae[a,:]
        larvaecost = larvaecost[a]
        
        # Each larva is assigned a place in the reef to settle
        nreef = np.random.permutation(P)
        nreef = nreef[0:Nlarvae]
        
        # larvae occupies empty places
        free = self.M * np.where(REEF == 0)[1] + np.where(REEF == 0)[0]
        free = np.intersect1d(free, nreef)
        
        REEF[(free%self.M).astype(int),(free/self.M).astype(int)] = 1
        REEFpob[free] = larvae[0:len(free)]
        REEFcost[free] = larvaecost[0:len(free)]
        
        larvae = np.delete(larvae, list(range(0,len(free))), axis=0)
        larvaecost = np.delete(larvaecost, list(range(0,len(free))), axis=0)
        
        # In the occupìed places there is a fight
        nreef = np.random.permutation(P)
        ocup = self.M * np.where(REEF == 1)[1] + np.where(REEF == 1)[0]
        ocup = np.intersect1d(ocup, nreef)
        Nlarvae = np.shape(larvae)[0]
        while Nlarvae != 0:
            k=k0
            ok=0
            while (k != 0) and (ok == 0):
                ind = np.random.randint(0,len(ocup))
                if (opt == 'max'):
                    if larvaecost[0] > REEFcost[ocup[ind]]:
                        # settle the larva
                        REEFpob[ocup[ind]] = larvae[0]
                        REEFcost[ocup[ind]] = larvaecost[0]
                        # eliminate the larva from larvae list
                        larvae = np.delete(larvae, 0, axis=0)
                        larvaecost = np.delete(larvaecost, 0, axis=0)
                        # eliminate the place from the occupied ones 
                        ocup = np.delete(ocup, ind)
                        ok = 1
                else:
                    print("Hay que realizar minimizacion")
                k -= 1
            if ok == 0: # Eliminate the larva
                larvae = np.delete(larvae, 0, axis=0)
                larvaecost = np.delete(larvaecost, 0, axis=0)
            Nlarvae = np.shape(larvae)[0]
                
        return REEF, REEFpob, REEFcost
        
    def Budding(self, REEF, REEFpob, REEFcost, Fa = None, opt = None):
        if Fa == None:
            Fa = self.Fa
        if opt == None:
            opt = self.opt
        
        pob = REEFpob[np.where(REEF==1)[0], :]
        
        REEFcost = REEFcost[self.M * np.where(REEF == 1)[1] + np.where(REEF == 1)[0]]
        N = pob.shape[0]
        NA = int(np.round(Fa*N))
        
        if opt == 'max':
            ind = np.argsort(-REEFcost)
        elif opt == 'min':
            ind = np.argsort(REEFcost)
            
        REEFcost = REEFcost[ind]
        Alarvae = pob[ind[0:NA], :]
        Acost = REEFcost[0:NA]
        
        return Alarvae, Acost
        
    def ExtremeDepredation(self, REEF, REEFpob, REEFcost, K = None):
        if K == None:
            K = self.K
        (U, indices, count) = np.unique(REEFpob, return_index=True, return_counts=True, axis=0) 
        if len(np.where(np.sum(U, axis= 1)==0)[0]) !=0:
            zero_ind = int(np.where(np.sum(U, axis= 1)==0)[0])
            indices = np.delete(indices, zero_ind)
            count = np.delete(count, zero_ind)

        while np.where(count>K)[0].size>0:
            higherk = np.where(count>K)[0]
            fil = (indices[higherk] % self.M).astype(int)
            col = (indices[higherk] / self.M).astype(int)
            REEF[fil,col] = 0
            REEFpob[indices[higherk], :] = 0
            REEFcost[indices[higherk]] = 0
            
            (U, indices, count) = np.unique(REEFpob, return_index=True, return_counts=True, axis=0) 
            if len(np.where(np.sum(U, axis= 1)==0)[0]) !=0:
                zero_ind = int(np.where(np.sum(U, axis= 1)==0)[0])
                indices = np.delete(indices, zero_ind)
                count   = np.delete(count, zero_ind)
        return REEF, REEFpob, REEFcost
    def Depredation(self, REEF, REEFpob, REEFcost, Fd = None, Pd = None, opt = None):
        if Fd == None:
            Fd = self.Fd
        if Pd == None:
            Pd = self.Pd
        if opt == None:
            opt = self.opt
            
        if opt == 'max':
            sortind = np.argsort(REEFcost)
            # sortfitness = REEFcost[sortind]
        elif opt == 'min':
            sortind = np.argsort(-REEFcost)
            # sortfitness = REEFcost[sortind] 
        ndep = round(Fd*(np.shape(REEFpob)[0]))
        sortind = sortind[0:ndep]
        
        p = np.random.rand(ndep)
        dep = np.where(p < Pd)[0]
        
        fil = (sortind[dep]%self.M).astype(int)
        col = (sortind[dep]/self.M).astype(int)
        REEF[fil,col] = 0
        REEFpob[sortind[dep],:] = 0
        REEFcost[sortind[dep]] = 0

        return REEF, REEFpob, REEFcost

    def StartCRO(self, Ngen = None):
        if Ngen == None:
            Ngen = self.Ngen
        
        self.REEF, self.REEFpob = self.ReefInitialization()
        self.REEFcost = self.ReefFitness(self.REEFpob)
        
        # Inicio bucle
        self.MejorCoste = np.zeros(Ngen)
        for i in range(Ngen):
            self.ESlarvae = self.BroadcastSpawning(self.REEF, self.REEFpob)
            self.ISlarvae = self.Brooding(self.REEF, self.REEFpob)
            
            self.ESlarvae = self.LarvaeCorrection(self.ESlarvae)
            self.ISlarvae = self.LarvaeCorrection(self.ISlarvae)
            self.larvae = np.concatenate([self.ESlarvae, self.ISlarvae])
            
            self.EScost = self.ReefFitness(self.ESlarvae)
            self.IScost = self.ReefFitness(self.ISlarvae)
            self.larvaecost = np.concatenate([self.EScost,self.IScost])
            
            self.REEF, self.REEFpob, self.REEFcost = self.LarvaeSetting(self.REEF, self.REEFpob, self.REEFcost, self.larvae, self.larvaecost)
            
            self.Alarvae, self.Acost = self.Budding(self.REEF, self.REEFpob, self.REEFcost)
            
            self.REEF, self.REEFpob, self.REEFcost = self.LarvaeSetting(self.REEF, self.REEFpob, self.REEFcost, self.Alarvae, self.Acost)
            
            if i != Ngen:
                self.REEF, self.REEFpob, self.REEFcost = self.ExtremeDepredation(self.REEF, self.REEFpob, self.REEFcost)
            
            self.ind = np.argsort(-self.REEFcost) # Maximization
            self.MejorCoste[i] = self.REEFcost[self.ind[0]]
            
        plt.figure()
        plt.title("Fitness Function")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.plot(self.MejorCoste)
        plt.show()   
        
def fitness(Individuo):
    Coste = sum(Individuo)
    return Coste


if __name__ == "__main__" :
    CRO_instance = CROClass()
    CRO_instance.ImprimirInformacion()
    
    # Puedes lanzar la aplicación utilizando el método StartCRO
    inicio = time.time()
    CRO_instance.StartCRO(Ngen = 200)
    fin = time.time()
    print("Tiempo utilizado = " + str(fin-inicio))
    
    # O puedes lanzar la aplicación en tu programa utilizando la siguiente estructura
    
    # REEF, REEFpob = CRO_instance.ReefInitialization()
    # REEFcost = CRO_instance.ReefFitness(REEFpob)
    
    # # Inicio bucle
    # Ngen = 500
    # MejorCoste = np.zeros(Ngen)
    # for i in range(Ngen):
    #     ESlarvae = CRO_instance.BroadcastSpawning(REEF, REEFpob)
    #     ISlarvae = CRO_instance.Brooding(REEF, REEFpob)
        
    #     ESlarvae = CRO_instance.LarvaeCorrection(ESlarvae)
    #     ISlarvae = CRO_instance.LarvaeCorrection(ISlarvae)
    #     larvae = np.concatenate([ESlarvae, ISlarvae])
        
    #     EScost = CRO_instance.ReefFitness(ESlarvae)
    #     IScost = CRO_instance.ReefFitness(ISlarvae)
    #     larvaecost = np.concatenate([EScost,IScost])
        
    #     REEF, REEFpob, REEFcost = CRO_instance.LarvaeSetting(REEF, REEFpob, REEFcost, larvae, larvaecost)
        
    #     Alarvae, Acost = CRO_instance.Budding(REEF, REEFpob, REEFcost)
        
    #     REEF, REEFpob, REEFcost = CRO_instance.LarvaeSetting(REEF, REEFpob, REEFcost, Alarvae, Acost)
        
    #     REEF, REEFpob, REEFcost = CRO_instance.ExtremeDepredation(REEF, REEFpob, REEFcost)
        
    #     ind = np.argsort(-REEFcost) # Maximization
    #     MejorCoste[i] = REEFcost[ind[0]]
        
    # plt.figure()
    # plt.plot(MejorCoste)
    # plt.show()
    # fin = time.time()
    # print("Tiempo utilizado = " + str(fin-inicio))
    
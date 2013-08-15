# TFTSim: Ternary Fission Trajectory Simulation in Python.
# Copyright (C) 2013 Patric Holmvall mail: patric.hol {at} gmail {dot} com
#
# TFTSim is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TFTSim is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with TFTSim.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division
import numpy as np
from scipy.stats import truncnorm
from TFTSim.tftsim_utils import *
from TFTSim.tftsim import *
import copy
from time import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.constants import codata
import pylab as pl
from sympy import Symbol
from sympy.solvers import nsolve
import sympy as sp
        
class CCTGenerator:
    """
    Generator for collinear configurations.
    
    """
    
    def __init__(self, sa, sims, mode, deltaDmin, deltaDmax, yMax, Dcount, ycount,
                 Ekin0 = 0, saveConfigs = False, oldConfigs = None):# Dmax, dx=0.5, yMax=0, dy=0, config='', Ekin0=0):
        """
        Initialize and pre-process the simulation data.

        :type sa: :class:`tftsim_args.TFTSimArgs` class
        :param sa: An instance of TFTSimArgs describing what kind of system to
                   simulate.
        
        :type sims: int
        :param sims: Number of simulations to run.
        
        """
        
        # Check for exceptions in input parameters

        self._sa = sa
        self._sims = sims
        if mode not in ["restUniform","uniform","uncertainty"]:
            raise ValueError("Selected mode doesn't exist. Valid modes: "+str(["restUniform","uniform","uncertainty"]))
        self._mode = mode
        self._saveConfigs = saveConfigs
        self._oldConfigs = oldConfigs
        if deltaDmin >= deltaDmax:
            raise ValueError("deltaDmin needs to be smaller than deltaDmax")
        self._deltaDmin = deltaDmin
        self._deltaDmax = deltaDmax
        self._yMax = yMax
        self._Dcount = Dcount
        self._ycount = ycount
        self._Ekin0 = Ekin0
        self._xcount = int(self._sims/(self._Dcount*self._ycount))
        #self._Dmax = Dmax
        #self._dx = dx
        #self._yMax = yMax
        #self._dy = dy
        #self._config = config
        #self._Ekin0 = Ekin0
        
        
        minTol = 0.0001

        # Calculate limits
        dcontact = self._sa.ab[0] + self._sa.ab[4]
        xcontact = self._sa.ab[0] + self._sa.ab[2]
        dsaddle = 1.0/(np.sqrt(self._sa.Z[1]/self._sa.Z[2])+1.0)
        xsaddle = 1.0/(np.sqrt(self._sa.Z[2]/self._sa.Z[1])+1.0)
        
        Eav = self._sa.Q - np.sum(self._Ekin0) - minTol
        
        # Solve Dmin
        Dsym = Symbol('Dsym')
        Dmin = np.float(nsolve(Dsym - (self._sa.Z[0]*self._sa.Z[1]/xsaddle + \
                                       self._sa.Z[0]*self._sa.Z[2]/dsaddle + \
                                       self._sa.Z[1]*self._sa.Z[2] \
                                      )*1.43996518/(Eav), Dsym, 18.0))
        
        # Limits for D in the simulation
        self._Dmin = Dmin + self._deltaDmin + minTol
        self._Dmax = Dmin + self._deltaDmax
        
        # What is minimum D when TP and LF are touching
        A = ((Eav)/1.43996518 - self._sa.Z[0]*self._sa.Z[2]/dcontact)/self._sa.Z[1]
        D_tpl_contact = np.float(nsolve(Dsym**2*A - \
                                        Dsym*(A*dcontact + self._sa.Z[0] + 
                                              self._sa.Z[2]) + \
                                        self._sa.Z[2]*dcontact, Dsym, 26.0))
        self._D_tpl_contact = D_tpl_contact
        # What is minimum D when TP and HF are touching
        A = ((Eav)/1.43996518 - self._sa.Z[0]*self._sa.Z[1]/xcontact)/self._sa.Z[2]
        D_tph_contact = np.float(nsolve(Dsym**2*A - \
                                        Dsym*(A*xcontact + self._sa.Z[0] + 
                                              self._sa.Z[1]) + \
                                        self._sa.Z[1]*xcontact, Dsym, 30.0))
        self._D_tph_contact = D_tph_contact

    def generate(self, filePath=None, plotInitialConfigs=False, verbose=False):
        """
        Generate initial configurations.
        """
        minTol = 0.0001
        if filePath == None:
            timeStamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
            filePath = "results/noname/" + str(timeStamp) + "/"
        
        # SimulatTrajectory object used to call checkConfig to verify that an
        # initial configuration is valid
        sim = SimulateTrajectory(sa=self._sa)
        
        if self._oldConfigs == None or not os.path.isfile(self._oldConfigs):
            if plotInitialConfigs:
                x_plot = [0]*self._sims
                y_plot = [0]*self._sims
                Ds = [0]*self._sims
                Ecs = [0]*self._sims
                        
            rs = np.zeros([self._sims,6])
            vs = np.zeros([self._sims,6])
            TXE = np.zeros(self._sims)


            D_linspace = np.linspace(self._Dmin, self._Dmax, self._Dcount)
            y_linspace = np.linspace(0.0, self._yMax, self._ycount)
            
            badOnes = 0
            s = 0
            for i in range(0,self._Dcount):
                D = D_linspace[i]
                
                for j in range(0,self._ycount):
                    y = y_linspace[j]
                    
                    # Calculate x-range
                    A = ((self._sa.Q-minTol-self._Ekin0)/1.43996518-self._sa.Z[1]*self._sa.Z[2]/D)/self._sa.Z[0]
                    p = [A,(self._sa.Z[2]-self._sa.Z[1]-A*D),D*self._sa.Z[1]]
                    sols = np.roots(p)
                    #print(str(D[i])+'\t'),
                    #print(sols),
                    
                    # Check 2 sols
                    if len(sols) != 2:
                        raise Exception('Wrong amount of solutions: '+str(len(sols)))

                    xmin = max(sols[1],self._sa.ab[0]+self._sa.ab[2]+minTol)
                    xmax = min(sols[0],D-(self._sa.ab[0]+self._sa.ab[4]+minTol))
                    
                    
                    x_linspace = np.linspace(xmin, xmax, self._xcount)
                    
                    for k in range(0,self._xcount):
                        initErrors = 0
                        x = x_linspace[k]
                    
                        # Start positions
                        r = [0,y,-x,0,D-x,0]
                        
                        #print(r)
                
                        Ec0 = self._sa.cint.coulombEnergies(Z_in=self._sa.Z,r_in=r,fissionType_in=self._sa.fissionType)
                        Eav = self._sa.Q - np.sum(Ec0)
                        # Get Center of Mass coordinates
                        xcm,ycm = getCentreOfMass(r_in=r, m_in=self._sa.mff)
                        rcm = [-xcm, y-ycm, -x-xcm, -ycm, (D-x)-xcm, -ycm]
                        
                        # Initial velocities            
                        v = [0.0] * 6
                
                        # Check that initial conditions are valid
                        initErrors += sim.checkConfiguration(r_in=r, v_in=v, TXE_in=0.0)
                        #sim = SimulateTrajectory(sa=self._sa, r_in=r, v_in=v, TXE=0.0)
                        #initErrors = sim.getExceptionCount()
                
                        if (np.sum(Ec0)) > self._sa.Q:
                            initErrors += 1
                        
                        # Store initial conditions if they are good
                        if initErrors > 0:
                            badOnes += 1    
                        rs[s] = r
                        vs[s] = v
                        
                        if plotInitialConfigs:
                            Ds[s] = D
                            x_plot[s] = x
                            y_plot[s] = y
                            Ecs[s] = np.sum(Ec0)
                        
                        s+=1
                        
                        if s%5000 == 0 and s > 0 and verbose:
                            print(str(s)+" of "+str(self._sims)+" initial "
                                  "conditions generated.")
                    # end of x loop
                # end of y loop
            # end of simulation while loop
            print("-----------------------------------------------------------")
            print(str(badOnes)+" out of "+str(self._sims)+" simulations failed.")
            
            # Save initial configurations
            if self._saveConfigs and (self._oldConfigs == None or not os.path.isfile(self._oldConfigs)):
                
                # Create results folder if it doesn't exist
                if not os.path.exists(filePath):
                    os.makedirs(filePath)
                
                # Store static variables in a shelved file format
                s = shelve.open(filePath + "initialConfigs.sb", protocol=pickle.HIGHEST_PROTOCOL)
                try:
                    s["0"] = { 'r0': rs, 'v0': vs, 'sims': self._sims}
                finally:
                    s.close()
                """
                f_data = file(filePath + "initialConfigs.bin", 'wb')
                np.save(f_data,np.array([rs, vs, TXE, self._sims]))
                f_data.close()
                """
                
                if verbose:
                    print("Saved initial configs to: "+ filePath + "initialConfigs.sb")
        else:
            sv = shelve.open(self._oldConfigs, protocol=pickle.HIGHEST_PROTOCOL)
            for row in sv:
                rs = sv[row]['r0']
                vs = sv[row]['v0']
                self._sims = sv[row]['sims']
                try:
                    TXE = sv[row]['TXE']
                except KeyError:
                    TXE = np.zeros(self._sims)
            """
            with open(self._oldConfigs,"rb") as f_data:
                npdata = np.load(f_data)
                rs = npdata[0]
                vs = npdata[1]
                TXE = npdata[2]
                self._sims = npdata[3]
            """
            Ecs = [0]*self._sims
            for ri in range(0,self._sims):
                Ecs[ri] = np.sum(self._sa.cint.coulombEnergies(Z_in=self._sa.Z,r_in=rs[ri],fissionType_in=self._sa.fissionType))
            if verbose:
                print("Loaded "+str(self._sims)+" intial configurations "
                      "from: "+self._oldConfigs)        
        if plotInitialConfigs:
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            ny, binsy, patches = ax.hist(Ecs, bins=100)
            bincentersy = 0.5*(binsy[1:]+binsy[:-1])
            l = ax.plot(bincentersy, ny, 'r--', linewidth=4,label='Ec0')
            ax.set_title('Initial Coulomb Energy, Q = %1.1f' % (self._sa.Q))
            ax.set_xlabel('Ec0 [MeV]')
            ax.set_ylabel('Counts')
            #ax.set_xlim([0,14.001])
            ax.legend()
            
            plt.show()
        
        return rs, vs, TXE, self._sims


        
    def oldShiz(self):
        minTol = 0.0001
        
        dcount = 0
        
        simulationNumber = 0
        totSims = 0
        
        if self._config == 'max':
            ekins = np.linspace(0,self._Ekin0,self._sims)
            D = []
            for i in range(0,self._sims):
                Eav = self._sa.Q - ekins[i] - minTol
                A = ((Eav)/1.43996518 - self._sa.Z[0]*self._sa.Z[1]/xcontact)/self._sa.Z[2]
                D_tph_contact = np.float(nsolve(Dsym**2*A - \
                                                Dsym*(A*xcontact + self._sa.Z[0] + 
                                                      self._sa.Z[1]) + \
                                                self._sa.Z[1]*xcontact, Dsym, 30.0))
                """Dmin = np.float(nsolve(Dsym - (self._sa.Z[0]*self._sa.Z[1]/xsaddle + \
                                               self._sa.Z[0]*self._sa.Z[2]/dsaddle + \
                                               self._sa.Z[1]*self._sa.Z[2] \
                                              )*1.43996518/(Eav), Dsym, 18.0))
                D.append(Dmin)
                xs.append([xsaddle*Dmin])"""
                D.append(D_tph_contact)
            xs = [[xcontact+minTol]]*self._sims
            ys = [0]*self._sims
            print D[0]
            print D[-1]
            ekins2 = np.linspace(0,192,self._sims)
            import matplotlib.pyplot as plt
            #plt.plot(ekins2, ekins2/(self._sa.mff[0]/self._sa.mff[1] + 1.0))
            #plt.plot(ekins2, ekins2/(self._sa.mff[1]/self._sa.mff[0] + 1.0))
            plt.plot(ekins,D)
            plt.show()
        elif self._config == 'triad':
            D = [Dmin, D_tpl_contact, D_tph_contact]
            xs = [[xsaddle*Dmin], [D[1]-dcontact-minTol], [xcontact+minTol]]
            ys = [0]*3
            print D
        else:
            D = np.linspace(Dmin, Dmax, self._sims)
        
            xs = [0]*self._sims
            ys = [0]*self._sims
            
            
            xs[0] = [xsaddle*D[0]]
            
            for i in range(1,len(D)):
                A = ((self._sa.Q-minTol)/1.43996518-self._sa.Z[1]*self._sa.Z[2]/D[i])/self._sa.Z[0]
                p = [A,(self._sa.Z[2]-self._sa.Z[1]-A*D[i]),D[i]*self._sa.Z[1]]
                sols = np.roots(p)
                #print(str(D[i])+'\t'),
                #print(sols),
                
                # Check 2 sols
                if len(sols) != 2:
                    raise Exception('Wrong amount of solutions: '+str(len(sols)))
                # Check reals
                #if np.iscomplex(sols[0]):
                #    raise ValueError('Complex root: '+str(sols[0]))
                #if np.iscomplex(sols[1]):
                #    raise ValueError('Complex root: '+str(sols[1]))
                
                xmin = max(sols[1],self._sa.ab[0]+self._sa.ab[2]+minTol)
                xmax = min(sols[0],D[i]-(self._sa.ab[0]+self._sa.ab[4]+minTol))
                
                """
                s1 = '_'
                s2 = '_'
                if xmin == self._sa.ab[0]+self._sa.ab[2]+minTol:
                    s1 = 'o'
                if xmax == D[i]-(self._sa.ab[0]+self._sa.ab[4]+minTol):
                    s2 = 'o'
                print(s1+s2)
                """
                
                # Check inside neck
                if xmin < (self._sa.ab[0]+self._sa.ab[2]):
                    raise ValueError('Solution overlapping with HF:')
                if xmin > D[i]-(self._sa.ab[0]+self._sa.ab[4]):
                    raise ValueError('Solution overlapping with LF:')
                if xmax < (self._sa.ab[0]+self._sa.ab[2]):
                    raise ValueError('Solution overlapping with HF:')
                if xmax > D[i]-(self._sa.ab[0]+self._sa.ab[4]):
                    raise ValueError('Solution overlapping with LF:')
                
                xs[i] = np.arange(xmin,xmax,self._dx)
        
        # Assign y-values
        for i in range(0,len(D)):
            ys[i] = [0]*len(xs[i])
            for j in range(0,len(xs[i])):
                if self._yMax == 0 or self._dy == 0:
                    ys[i][j] = [0]
                else:
                    ys[i][j] = np.arange(0,self._yMax,self._dy)
        
        # Count total simulations
        for i in range(0,len(D)):
            for j in range(0,len(xs[i])):
                    totSims += len(ys[i][j])
        
        ENi_max = 0
        for i in range(0,len(D)):
            for j in range(0,len(xs[i])):
                for k in range(0,len(ys[i][j])):
                    simulationNumber += 1
                    r = [0.0,ys[i][j][k],-xs[i][j],0.0,D[i]-xs[i][j],0.0]
                    
                    v = [0.0]*6
                    
                    if self._config == 'max':
                        v[0] = np.sqrt(2*ekins[i]/(self._sa.mff[0]**2/self._sa.mff[1] + self._sa.mff[0]))
                        v[2] = -np.sqrt(2*ekins[i]/(self._sa.mff[1]**2/self._sa.mff[0] + self._sa.mff[1]))
                    
                    sim = SimulateTrajectory(sa=self._sa, r_in=r, v_in=v)
                    e, outString, ENi = sim.run(simulationNumber=simulationNumber, timeStamp=timeStamp)
                    if ENi > ENi_max:
                        ENi_max = ENi
                    #sim.plotTrajectories()
                    if e == 0:
                        print("S: "+str(simulationNumber)+"/~"+str(totSims)+"\t"+str(r)+"\t"+outString)
        print("Total simulation time: "+str(time()-simTime)+"sec")
        print("Maximum Nickel Energy: "+str(ENi_max)+" MeV")


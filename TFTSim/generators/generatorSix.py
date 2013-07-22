# This file is part of TFTSim.
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
        
class GeneratorSix:
    """
    Generator for collinear configurations.
    
    """
    
    def __init__(self, sa, sims, Dmax, dx=0.5, yMax=0, dy=0, config='', Ekin0=0):
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
        self._Dmax = Dmax
        self._dx = dx
        self._yMax = yMax
        self._dy = dy
        self._config = config
        self._Ekin0 = Ekin0
        

    def generate(self):
        """
        Generate initial configurations.
        """  
        
        minTol = 0.0001
        
        dcount = 0
        simTime = time()
        timeStamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
        
        simulationNumber = 0
        totSims = 0
        
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
        Dmax = Dmin + self._Dmax
        
        # Solve D_tpl_contact and D_tph_contact
        A = ((Eav)/1.43996518 - self._sa.Z[0]*self._sa.Z[2]/dcontact)/self._sa.Z[1]
        D_tpl_contact = np.float(nsolve(Dsym**2*A - \
                                        Dsym*(A*dcontact + self._sa.Z[0] + 
                                              self._sa.Z[2]) + \
                                        self._sa.Z[2]*dcontact, Dsym, 26.0))
        A = ((Eav)/1.43996518 - self._sa.Z[0]*self._sa.Z[1]/xcontact)/self._sa.Z[2]
        D_tph_contact = np.float(nsolve(Dsym**2*A - \
                                        Dsym*(A*xcontact + self._sa.Z[0] + 
                                              self._sa.Z[1]) + \
                                        self._sa.Z[1]*xcontact, Dsym, 30.0))
        
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


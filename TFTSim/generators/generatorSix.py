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
    
    def __init__(self, sa, sims):
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
        

    def generate(self):
        """
        Generate initial configurations.
        """  
        
        dcount = 0
        simTime = time()
        timeStamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
        
        simulationNumber = 0
        
        dcontact = self._sa.ab[0]+self._sa.ab[4]
        dsaddle = 1.0/(np.sqrt(self._sa.Z[1]/self._sa.Z[2])+1.0)
        xsaddle = 1.0/(np.sqrt(self._sa.Z[2]/self._sa.Z[1])+1.0)
        
        
        # Solve Dmin
        Dsym = Symbol('Dsym')
        Dmin = np.float(nsolve(Dsym - (self._sa.Z[0]*self._sa.Z[1]/xsaddle + \
                                       self._sa.Z[0]*self._sa.Z[2]/dsaddle + \
                                       self._sa.Z[1]*self._sa.Z[2] \
                                      )*1.43996518/(self._sa.Q), Dsym, 18.0))
        
        # Solve Dmax
        #A = (self._sa.Q/1.43996518 - self._sa.Z[0]*self._sa.Z[2]/dcontact)/self._sa.Z[1]
        #Dmax = np.float(nsolve(Dsym - (Dsym**2*A - \
        #                               Dsym*(A*dcontact + self._sa.Z[2] + \
        #                                  self._sa.Z[0]) - \
        #                               A*dcontact + self._sa.Z[2]*dcontact
        #                              ), Dsym, 26.0))
        Dmax = Dmin+10.0
        
        print('Dmin = '+str(Dmin))
        print('Dmax = '+str(Dmax))
        
        D = np.linspace(Dmin, Dmax, self._sims)
        totSims = 0
        
        for i in range(0,self._sims):
            A = (self._sa.Q/1.43996518-self._sa.Z[1]*self._sa.Z[2]/D[i])/self._sa.Z[0]
            p = [A,(self._sa.Z[2]-self._sa.Z[1]-A*D[i]),D[i]*(self._sa.Z[1]-A)]
            sols = np.roots(p)
            print(sols),
            
            # Check 2 sols
            if len(sols) != 2:
                raise Exception('Wrong amount of solutions: '+str(len(sols)))
            # Check reals
            if np.iscomplex(sols[0]):
                raise ValueError('Complex root: '+str(sols[0]))
            if np.iscomplex(sols[1]):
                raise ValueError('Complex root: '+str(sols[1]))
            
            xmin = max(sols[1],self._sa.ab[0]+self._sa.ab[2])
            xmax = min(sols[0],D[i]-(self._sa.ab[0]+self._sa.ab[4]))
            
            s1 = '_'
            s2 = '_'
            if xmin == self._sa.ab[0]+self._sa.ab[2]:
                s1 = 'o'
            if xmax == D[i]-(self._sa.ab[0]+self._sa.ab[4]):
                s2 = 'o'
            
            print(s1+s2)
            
            # Check inside neck
            if xmin < (self._sa.ab[0]+self._sa.ab[2]):
                raise ValueError('Solution overlapping with HF:')
            if xmin > D[i]-(self._sa.ab[0]+self._sa.ab[4]):
                raise ValueError('Solution overlapping with LF:')
            if xmax < (self._sa.ab[0]+self._sa.ab[2]):
                raise ValueError('Solution overlapping with HF:')
            if xmax > D[i]-(self._sa.ab[0]+self._sa.ab[4]):
                raise ValueError('Solution overlapping with LF:')
            
            """
            for j in range(0,len(ds)):
                simulationNumber += 1
                r = [0.0,0.0,ds[j]-D[i],0.0,ds[j],0.0]
                v = [0.0]*6
                
                sim = SimulateTrajectory(sa=self._sa, r_in=r, v_in=v)
                e, outString = sim.run(simulationNumber=simulationNumber, timeStamp=timeStamp)
                #sim.plotTrajectories()
                if e == 0:
                    print("S: "+str(simulationNumber)+"/~"+str(totSims)+"\t"+str(r)+"\t"+outString)"""
        print("Total simulation time: "+str(time()-simTime)+"sec")


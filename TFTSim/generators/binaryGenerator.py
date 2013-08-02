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
        
class BinaryGenerator:
    """
    Generator for binary fission.
    """
    
    def __init__(self, sa, sims, Dmin, Dmax, Ekin0):
        """
        Initialize and pre-process the simulation data.

        :type sa: :class:`tftsim_args.TFTSimArgs` class
        :param sa: An instance of TFTSimArgs describing what kind of system to
                   simulate.
        
        :type sims: int
        :param sims: Number of simulations to run.
        
        :type Dmin: float
        :param Dmin: D value to start simulations on.
        
        :type Dmax: float
        :param Dmax: D value to stop simulations on.
         
        :type Ekin0: float
        :param Ekin0: Initial kinetic energy.
        """
        
        # Check for exceptions in input parameters

        self._sa = sa
        self._sims = sims
        self._Dmin = Dmin
        self._Dmax = Dmax
        self._Ekin0 = Ekin0
        
        self._minTol = 0.1

        self._Q = sa.Q
        
        # Failsafes that makes returns an exception if this procedure is incompatible with the fissioning system 
        #if derp:
        #   raise Exception('herp')
       
        
    def generate(self):
        """
        Generate initial configurations.
        """  
        
        dcount = 0
        simTime = time()
        timeStamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
        
        Ds = np.linspace(self._Dmin, self._Dmax, self._sims)
        
        simulationNumber = 0
        
        for D in Ds:
            simulationNumber += 1
            r = [0,0,D,0]
            
            Ec0 = self._sa.cint.coulombEnergies(Z_in=self._sa.Z, r_in=r,fissionType_in=self._sa.fissionType)
            Eav = self._Q - np.sum(Ec0)
            
            v1 = 0
            v2 = 0
            
            # Randomize a kinetic energy for the ternary particle
            #Ekin_tot = np.random.uniform(0.0,Eav*0.9)
            
            # Randomize initial direction for the initial momentum of tp
            """
            p2 = Ekin_tot * 2 * self._sa.mff[0]
            px2 = np.random.uniform(0.0,p2)
            py2 = p2 - px2
            xdir = np.random.randint(2)*2 - 1
            ydir = np.random.randint(2)*2 - 1
            """
            
            #v = [xdir*np.sqrt(px2)/self._sa.mff[0],np.sqrt(py2)/self._sa.mff[0],0.0,0.0,0.0,0.0]
            v = [v1,0,v2,0]
            
            sim = SimulateTrajectory(sa=self._sa, r_in=r, v_in=v)
            e, outString = sim.run(simulationNumber=simulationNumber, timeStamp=timeStamp)
            
            if e == 0:
                print("S: "+str(simulationNumber)+"/~"+str(self._sims)+"\t"+str(r)+"\t"+outString)
        print("Total simulation time: "+str(time()-simTime)+"sec")


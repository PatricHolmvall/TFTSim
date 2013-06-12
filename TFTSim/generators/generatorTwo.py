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

class GeneratorTwo:
    """
    A second, less naive, attempt to generate a lot of starting configurations.
    This uses the same basic configuration as generatorOne, but with statistical
    distribution of D, x and y.
    
    First, randomize x uniformly between 0 and 1. 0 means tp is basically
    touching hf, and 1 means tp is touching lf. After that, randomize y close to
    the fission axis with a probability density that rapidly decreases as you
    get further from the axis, possibly with some mininum value ymin as we might
    not want to study cases where the ternary particle is on-axis.
    After that, randomize D close to Dmin with rapidly decreasing probability
    as you move further away from Dmin. Dmin is solved numerically from the
    equation Q = E_coulomb.
    
    """
    
    def __init__(self, sa, sims, DMu, DSigma, yMu, ySigma, ymin):
        """
        Initialize and pre-process the simulation data.

        :type sa: :class:`tftsim_args.TFTSimArgs` class
        :param sa: An instance of TFTSimArgs describing what kind of system to
                   simulate.
        
        :type sims: int
        :param sims: Number of simulations to run.
        
        :type DMu: float
        :param DMu: Center of the gaussian probability for D, measured from
                    Dmin in fm.
        
        :type DSigma: float
        :param DSigma: Spread of the gaussian probability for D in fm,
                       corresponding to the FWHM.
        
        :type yMu: float
        :param yMu: Center of the gaussian probability for y. Measured from
                    ymin in fm.
        
        :type ySigma: float
        :param ySigma: Spread of the gaussian probability for y in fm,
                       corresponding to the FWHM.
        
        :type ymin: float
        :param ymin: Minimum displacement of ternary particle from fission axis
                     in fm. A value of zero is generally not considered.
        """
        
        # Check for exceptions in input parameters

        self._sa = sa
        self._sims = sims
        self._DMu = DMu
        self._DSigma = DSigma
        self._yMu = yMu
        self._ySigma = ySigma
        self._ymin = ymin
        self._Z = [self._sa.tp.Z, self._sa.hf.Z, self._sa.lf.Z]
        self._rad = [crudeNuclearRadius(self._sa.tp.A),
                     crudeNuclearRadius(self._sa.hf.A),
                     crudeNuclearRadius(self._sa.lf.A)]
        self._mff = [u2m(self._sa.tp.A), u2m(self._sa.hf.A), u2m(self._sa.lf.A)]
        self._minTol = 0.1

        self._Q = getQValue(self._sa.fp.mEx,self._sa.pp.mEx,self._sa.tp.mEx,self._sa.hf.mEx,self._sa.lf.mEx,self._sa.lostNeutrons)

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
        for simulationNumber in range(1,self._sims+1):
            # Randomize x
            #xr = np.random.uniform(0.0, 1.0)
            """
            if np.random.uniform(0.0,1.0) <= 0.4:
                xr = np.random.uniform(0.0,0.5)
            else:
                xr = np.random.uniform(0.5,1.0)
            """
            xr = np.random.uniform(0.0,1.0)
            # Randomize y
            #y = np.random.lognormal(self._yMu, self._ySigma) + self._ymin
            y = np.random.uniform(self._ymin,6.0)
            #y = np.random.gamma(self._yMu, self._ySigma) + self._ymin
            # Randomize D
            Dmin = self._sa.pint.solveD(xr,y,self._Q,self._Z,self._rad,sol_guess=18.0)
            #Dmin = min(np.float(solveDmin(xr, y)), )
            #D = np.random.lognormal(self._DMu, self._DSigma) + Dmin
            D = np.random.gamma(self._DMu, self._DSigma) + Dmin

            x = D*xr + self._rad[0] + self._rad[1] - xr*(2*self._rad[0] + self._rad[1] + self._rad[2])
            
            r = [0.0,y,-x,0.0,D-x,0.0]
            
            Ec0 = self._sa.pint.coulombEnergies(self._Z, r)
            Eav = self._Q - np.sum(Ec0)
            
            # Randomize a kinetic energy for the ternary particle
            Ekin_tot = np.random.uniform(0.0,Eav*0.9)
            
            # Randomize initial direction for the initial momentum of tp
            p2 = Ekin_tot * 2 * self._mff[0]
            px2 = np.random.uniform(0.0,p2)
            py2 = p2 - px2
            xdir = np.random.randint(2)*2 - 1
            ydir = np.random.randint(2)*2 - 1
            
            #v = [xdir*np.sqrt(px2)/self._mff[0],np.sqrt(py2)/self._mff[0],0.0,0.0,0.0,0.0]
            v = [0.0,0.0,0.0,0.0,0.0,0.0]
            
            sim = SimulateTrajectory(sa=self._sa, r=r, v=v)
            e, outString = sim.run(simulationNumber=simulationNumber, timeStamp=timeStamp)
            if e == 0:
                print("S: "+str(simulationNumber)+"/"+str(self._sims)+"\t"+str(r)+"\t"+outString)
        print("Total simulation time: "+str(time()-simTime)+"sec")


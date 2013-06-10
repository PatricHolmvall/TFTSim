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
from sympy import Symbol
from sympy.solvers import nsolve
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
        #self._xShape = xShape
        self._yMu = yMu
        self._ySigma = ySigma
        self._ymin = ymin
        self._rtp = crudeNuclearRadius(self._sa.tp.A)
        self._rhf = crudeNuclearRadius(self._sa.hf.A)
        self._rlf = crudeNuclearRadius(self._sa.lf.A)
        self._mff = [u2m(self._sa.tp.A), u2m(self._sa.hf.A), u2m(self._sa.lf.A)]
        self._minTol = 0.1

        self._ke2 = 1.439964
        self._Q = getQValue(self._sa.fp.mEx,self._sa.pp.mEx,self._sa.tp.mEx,self._sa.hf.mEx,self._sa.lf.mEx,self._sa.lostNeutrons)

        """
        xmin = (self._rtp + self._rhf)
        E12_max = self._ke2*np.float(self._sa.tp.Z*self._sa.hf.Z) / (xmin)
        Eav = self._Q - E12_max
        
        A = xmin + self._ke2 * self._sa.lf.Z*(self._sa.hf.Z+self._sa.tp.Z) / Eav
        B = (xmin * self._ke2 * self._sa.hf.Z * self._sa.lf.Z) / Eav
        self._Dmin = 0.5 * A + np.sqrt(0.25*A**2 - B)
        
        self._xmin = self._rtp + self._rhf + self._minTol
        """
        
        # Failsafes that makes returns an exception if this procedure is incompatible with the fissioning system 
        #if derp:
        #   raise Exception('herp')

    def generate(self):
        """
        Generate initial configurations.
        """
        def solveDmin(x, y):
            a = (self._sa.tp.Z*self._sa.hf.Z)
            b = (self._sa.tp.Z*self._sa.lf.Z)
            c = (self._sa.hf.Z*self._sa.lf.Z)
            A = self._rhf + self._rtp - x*(2*self._rtp + self._rhf + self._rlf)
            
            dmin = Symbol('dmin')
            return nsolve(a/((dmin*x+A)**2+y**2)**(0.5) + \
                          b/((dmin*(1-x)-A)**2+y**2)**(0.5) + \
                          c/dmin-self._Q/self._ke2, dmin, 18.0)   
        
        
        dcount = 0
        simTime = time()
        timeStamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
        for simulationNumber in range(1,self._sims+1):
            # Randomize x
            #xp = np.random.uniform(0.0, 1.0)
            """
            if np.random.uniform(0.0,1.0) <= 0.4:
                xp = np.random.uniform(0.0,0.5)
            else:
                xp = np.random.uniform(0.5,1.0)
            """
            xp = np.random.uniform(0.0,1.0)
            # Randomize y
            #y = np.random.lognormal(self._yMu, self._ySigma) + self._ymin
            y = np.random.gamma(self._yMu, self._ySigma) + self._ymin
            # Randomize D
            Dmin = np.float(solveDmin(xp, y))
            #Dmin = min(np.float(solveDmin(xp, y)), )
            #D = np.random.lognormal(self._DMu, self._DSigma) + Dmin
            D = np.random.gamma(self._DMu, self._DSigma) + Dmin

            x = D*xp + self._rhf + self._rtp - xp*(2*self._rtp + self._rhf + self._rlf)
            
            r = [0.0,y,-x,0.0,D-x,0.0]
            
            Ec0 = self._sa.pint.coulombEnergies([self._sa.tp.Z,self._sa.hf.Z,self._sa.lf.Z], r)
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
        
        """
        for D in np.arange(self._Dmin, self._Dmax, self._Dinc):
            dcount += 1
            self._xmax = D - (self._rtp+self._rlf) - minTol
            for x in np.arange(self._xmin, self._xmax, self._xinc):
                for y in np.arange(self._ymin, self._ymax, self._yinc):
                    simulationNumber += 1
                    r = [0.0,y,-x,0.0,D-x,0.0]
                    sim = SimulateTrajectory(sa=self._sa, r=r)
                    e, outString = sim.run(simulationNumber=simulationNumber, timeStamp=timeStamp)
                    if e == 0:
                        print("D: "+str(dcount)+"/"+str(len(np.arange(self._Dmin, self._Dmax, self._Dinc)))+
                              "\tS(D):"+str(len(np.arange(self._xmin, self._xmax, self._xinc))*\
                                            len(np.arange(self._ymin, self._ymax, self._yinc)))+
                              "\t"+str(r)+"\t"+outString)
        """
        print("Total simulation time: "+str(time()-simTime)+"sec")


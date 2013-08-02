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
from TFTSim.tftsim_utils import *
from TFTSim.tftsim import *
import copy
from time import time
from datetime import datetime

class GeneratorOne:
    """
    Randomly generate initial configurations. The parametrization goes like:
    D - the initial distance between hf and lf along the fission axis.
    x - the x distance between hf and tp.
    y - the y distance between hf and tp.
    Origo is put on the fission axis exactly below tp, so that
    r_init = [xtp=0, ytp=y, xhf=-x, yhf=0, xlf=D-x, ylf=0]
    
    """
    
    def __init__(self, sa, Dmax, Dinc, xinc, yinc, ymax, ymin):
        """
        Pre-process and initialize the simulation data.

        :type sa: :class:`tftsim_args.TFTSimArgs` class
        :param sa: An instance of TFTSimArgs describing what kind of system to
                   simulate.
                   
        :type Dmax: float
        :param Dmax: Maximum separation of fission fragments in fm.
        
        :type Dinc: float
        :param Dinc: Step size of each increment of Dinc in fm.
        
        :type yinc: float
        :param yinc: Step size of each increment of yinc in fm.
        
        :type ymax: float
        :param ymax: Maximum displacement of ternary particle from fission axis
                     in fm.
        
        :type ymin: float
        :param ymin: Minimum displacement of ternary particle from fission axis
                     in fm. A value of zero is generally not considered.
        """

        self._sa = sa
        self._Z = [self._sa.tp.Z, self._sa.hf.Z, self._sa.lf.Z]
        self._rad = [crudeNuclearRadius(self._sa.tp.A),
                     crudeNuclearRadius(self._sa.hf.A),
                     crudeNuclearRadius(self._sa.lf.A)]
        self._mff = [u2m(self._sa.tp.A), u2m(self._sa.hf.A), u2m(self._sa.lf.A)]
        minTol = 0.1
 
        ke2 = 1.439964
        self._Q = getQValue(self._sa.fp.mEx,self._sa.pp.mEx,self._sa.tp.mEx,self._sa.hf.mEx,self._sa.lf.mEx,self._sa.lostNeutrons)

        self._xmin = self._rad[0] + self._rad[1]
        
        E12_max = self._sa.pint.coulombEnergy(self._Z[0:2],[0,self._xmin])
        Eav = self._Q - E12_max
        
        A = self._xmin + ke2*self._sa.lf.Z*(self._sa.hf.Z+self._sa.tp.Z) / Eav
        B = (self._xmin * ke2 * self._sa.hf.Z * self._sa.lf.Z) / Eav
        self._Dmin = 0.5 * A + np.sqrt(0.25*A**2 - B)
        
        self._Dmax = self._Dmin + Dmax
        self._Dinc = Dinc

        self._xinc = xinc
        self._ymin = ymin
        self._ymax = ymax
        self._yinc = yinc
        
        # Failsafes that makes returns an exception if this procedure is incompatible with the fissioning system 
        #if derp:
        #   raise Exception('herp')
        
    def generate(self):
        """
        Generate initial configurations.
        """
        
        minTol = 0.1
        simulationNumber = 0
        dcount = 0
        simTime = time()
        timeStamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
        for D in np.arange(self._Dmin, self._Dmax, self._Dinc):
            dcount += 1
            self._xmax = D - (self._rtp+self._rlf) - minTol
            for x in np.arange(self._xmin, self._xmax, self._xinc):
                for y in np.arange(self._ymin, self._ymax, self._yinc):
                    simulationNumber += 1
                    r = [0.0,y,-x,0.0,D-x,0.0]
                    v = [0.0] * 6
                    sim = SimulateTrajectory(sa=self._sa, r=r, v=v)
                    e, outString = sim.run(simulationNumber=simulationNumber, timeStamp=timeStamp)
                    if e == 0:
                        print("S: "+str(simulationNumber)+"/"+str(self._sims)+"\t"+str(r)+"\t"+outString)
        print("Total simulation time: "+str(time()-simTime)+"sec")


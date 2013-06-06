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
from TFTSim.tftsim_utils import *
from TFTSim.tftsim import *
import copy
from time import time
from datetime import datetime

class GeneratorOne:
    """
    A first, naive, attempt to generate a lot of starting configurations. The
    generator starts with a minimum distance (D) between the heavy (HF) and
    light (LF) fission fragments (FF). This is calculated from the Q-value of
    the reaction. The distance D will typically be one or a few fermi meters
    (fm) larger than this since the ternary particle (TP) will contribute to the
    potential energy as well.
    
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
        self._rtp = crudeNuclearRadius(self._sa.tp.A)
        self._rhf = crudeNuclearRadius(self._sa.hf.A)
        self._rlf = crudeNuclearRadius(self._sa.lf.A)
        minTol = 0.1
 
        ke2 = 1.439964
        self._Q = getQValue(self._sa.fp.mEx,self._sa.pp.mEx,self._sa.tp.mEx,self._sa.hf.mEx,self._sa.lf.mEx,self._sa.lostNeutrons)

        self._xmin = self._rtp + self._rhf
        E12_max = ke2*np.float(self._sa.tp.Z*self._sa.hf.Z) / (self._xmin)
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
                    sim = SimulateTrajectory(sa=self._sa, r=r)
                    e, outString = sim.run(simulationNumber=simulationNumber, timeStamp=timeStamp)
                    if e == 0:
                        print("D: "+str(dcount)+"/"+str(len(np.arange(self._Dmin, self._Dmax, self._Dinc)))+
                              "\tS(D):"+str(len(np.arange(self._xmin, self._xmax, self._xinc))*\
                                            len(np.arange(self._ymin, self._ymax, self._yinc)))+
                              "\t"+str(r)+"\t"+outString)
        print simulationNumber



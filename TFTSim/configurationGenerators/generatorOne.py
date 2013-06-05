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

class GeneratorOne:
    """
    A first, naive, attempt to generate a lot of starting configurations. The
    generator starts with a minimum distance (D) between the heavy (HF) and
    light (LF) fission fragments (FF). This is calculated from the Q-value of
    the reaction. The distance D will typically be one or a few fermi meters
    (fm) larger than this since the ternary particle (TP) will contribute to the
    potential energy as well.
    
    Uses a Coulomb interaction between particles were each particle is
    modelled as a point, i.e. |F| = k*q1*q2/|r12^2|.

    Contains the equation of motion a = k*q1*q2*r12/(m*|r12^3|) and the Coulomb
    Energy E = -k*q1*q2/|r12|.
    """
    
    def __init__(self, pint, fp, pp, tp, hf, lf, lostNeutrons):
        """
        
        """
        Q = 185.891
        ke2 = 1.439964
        Ztp = 2
        Zhf = 52
        Zlf = 38
        rtp = 1.98425
        rhf = 6.3965
        rl = 5.72357
        E12_max = ke2*np.float(tp.Z*hf.Z) / (rtp+rhf)
        Eav = Q - E12_max
        
        D = (Eav/(ke*lf.Z) + (rhf+rtp)*hf.Z) / (hf.Z + tp.Z)


    # Failsafes that makes returns an exception if this procedure is incompatible with the fissioning system 
 

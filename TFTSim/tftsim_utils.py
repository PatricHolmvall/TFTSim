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
import sys
import numpy as np
import math

"""
Collection of functions frequently used by different parts of TFTSim.
"""

def u2m(m):
    """
    Converts a mass given in (unified) atomic mass units (u) to a mass given in
    (electron Volt)/(speed of light)^2 (eV/c^2)
    through the simple relation 1u = 931.494061(21) MeV/c^2 (Source: wikipedia)
    
    :type m: float
    :param m: Mass of particle in atomic mass units.

    :rtype: float
    :returns: Mass of particle in MeV/c^2.
    """
    return np.float(m) * 931.494061

def crudeNuclearRadius(A, r0=1.25):
    """
    Calculate the nuclear radius of a particle of atomic mass number A through
    the crude approximation r = r0*A^(1/3), where [r0] = fm.
    
    :type A: int
    :parameter A: Atomic mass number.
    
    :type r0: float
    :parameter r0: Radius coefficient that you might want to vary as A varies
                   to get the correct fit.
    
    :rtype: float
    :returns: Nuclear radius in fm.
    """
    return r0 * (np.float(A)**(1.0/3.0))

def getQValue(mEx_fp, mEx_pp, mEx_tp, mEx_hf, mEx_lf, lostNeutrons):
    """
    Calculate the Q value of a given fission process.
    
    :type mEx_fp: float
    :param mEx_fp: Excess mass of fissioning particle (needs to be positive).
    
    :type mEx_pp: float
    :param mEx_pp: Excess mass of projectile particle (needs to be positive).
    
    :type mEx_tp: float
    :param mEx_tp: Excess mass of ternary particle.
    
    :type mEx_hf: float
    :param mEx_hf: Excess mass of heavy fission fragment.
    
    :type mEx_lf: float
    :param mEx_lf: Excess mass of light fission fragment.
    
    :type lostNeutrons: int
    :param lostNeutrons: Number of lost neutrons in the fission process.
    
    :rtype: float
    :returns: Q-value for a given fission process, in MeV/c^2.
    """
    
    mEx_neutron = 8.071 # Excess mass of the neutron in MeV/c^2
    
    return np.float(mEx_fp + mEx_pp - mEx_tp - mEx_hf - mEx_lf -
                    lostNeutrons*mEx_neutron) 

def getKineticEnergy(m,v):
    """
    Retruns kinetic energy of a non-relativistic particle.

    :type m: float
    :param m: Mass of the particle
    
    :type v: list of floats
    :param v: Velocity in x and y of the particle.

    :rtype: float
    :returns: Kinetic energy (E=m*v^2/2).
    """
    return m*(v[0]**2+v[1]**2)*0.5

def getDistance(r1,r2):
    """
    Get distance between two vectors.
    
    :type r1: list of floats
    :param r1: x- and y-coordinate of first vector.
    
    :type r2: list of floats
    :param r2: x- and y-coordinate of second vector.
    
    :rtype: float
    :returns: Distance between two vectors.
    """
    
    return np.sqrt((r1[0]-r2[0])**2+(r1[1]-r2[1])**2)

def getAngle(r1,r2):
    """
    Get angle between two vectors.
    
    :type r1: list of floats
    :param r1: x- and y-coordinate of first vector.
    
    :type r2: list of floats
    :param r2: x- and y-coordinate of second vector.
    
    :rtype: float
    :returns: Angle between two vectors.
    """
    
    inner_product = r1[0]*r2[0] + r1[1]*r2[1]
    len1 = math.hypot(r1[0], r1[1])
    len2 = math.hypot(r2[0], r2[1])
    return math.acos(inner_product/(len1*len2))*180.0/np.pi

def humanReadableSize(size):
    """
    Converts a size in bytes to human readable form.

    :param number size: Size in bytes to convert

    :rtype: string
    :returns: The size in human readable form.
    """
    if size <= 0 or size >= 10e18:
        return "%.3g" % size + " B"
    else:
        p = int(np.log(size) / np.log(1024))
        names = ["", "ki", "Mi", "Gi", "Ti", "Pi", "Ei"]
        return "%.3g" % (size / 1024.0 ** p) + " " + names[p] + "B"


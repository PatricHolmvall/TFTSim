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
import matplotlib.pyplot as plt
from scipy.constants import codata

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
    #nc = codata.value('speed of light in vacuum')**2 * 1e-16
    #return np.float(m) * codata.value('atomic mass constant energy equivalent in MeV') / (nc * 1e4)
    return np.float(m) * codata.value('atomic mass constant energy equivalent in MeV')


def crudeNuclearRadius(A_in, r0=1.25):
    """
    Calculate the nuclear radius of a particle of atomic mass number A through
    the crude approximation r = r0*A^(1/3), where [r0] = fm.
    
    :type A_in: int
    :parameter A_in: Atomic mass number.
    
    :type r0: float
    :parameter r0: Radius coefficient that you might want to vary as A varies
                   to get the correct fit.
    
    :rtype: float
    :returns: Nuclear radius in fm.
    """
    
    #if A > 40:
    #    r0 = 1.16
    #else:
    #    r0 = 1.22
    #
    # r0 = 0.94 + 32.0/(Z_in**2 + 200.0)
    #4*Pi*0.95*(1-1.7826*((134-2*52)/134)^2)*((1-(1/(1.2*(134)^(1/3)))^2)^(-1)+(1-(1/(1.2*(96)^(1/3)))^2)^(-1))*(-4.41*e^(-9/0.7176))
    
    return r0 * (np.float(A_in)**(1.0/3.0))


def getQValue(mEx_pre_fission, mEx_post_fission, lostNeutrons):
    """
    Calculate the Q value of a given fission process.
    
    :type mEx_pre_fission: float
    :param mEx_pre_fission: Excess mass of fissioning system.
    
    :type mEx_post_fission: float
    :param mEx_post_fission: Excess mass of fission products.
    
    :type lostNeutrons: int
    :param lostNeutrons: Number of lost neutrons in the fission process.
    
    :rtype: float
    :returns: Q-value for a given fission process, in MeV/c^2.
    """
    
    mEx_neutron = 8.071 # Excess mass of the neutron in MeV/c^2
    
    return mEx_pre_fission - mEx_post_fission - \
           np.float(lostNeutrons*mEx_neutron) 


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


def circleEllipseOverlap(r_in, a_in, b_in, rad_in):
    """
    Assert if a circle and an ellipse overlap.
    
    :type r_in: list of floats
    :param r_in: Coordinates of center of the circle and ellipse:
                 r=[x_circle, y_circle, x_ellipse, y_ellipse].
                 
    :type a_in: float
    :param a_in: Semimajor axis of the ellipse.
    
    :type b_in: float
    :param b_in: Semiminor axis of the ellipse.
    
    :type rad_in: float
    :param rad_in: Radius of the circle.
    
    :rtype: boolean
    :returns: True if the circle and ellipse overlap, False otherwise.
    """
    
    return (r_in[2]-r_in[0])**2/(a_in+rad_in)**2 + \
           (r_in[3]-r_in[1])**2/(b_in+rad_in)**2 <= 1


def plotEllipse(x0_in,y0_in,a_in,b_in):#,color_in,lineStyle_in,lineWidth_in):
    phi = np.linspace(0.0,2*np.pi,100)
    na=np.newaxis
    x_line = x0_in + a_in*np.cos(phi[:,na])
    y_line = y0_in + b_in*np.sin(phi[:,na])
    plt.plot(x_line,y_line,'b-', linewidth=3.0)
    
    
def getClosestConfigurationLine(D_in,Dsize_in,E_in,Z_in,pint_in,ab_in):
    """
    xl = np.linspace(0.0,D_in,500)
    ylQ = np.zeros_like(xl)
    ylQf = np.zeros_like(xl)
    for i in range(0,len(ylQ)):
        ylQ[i] = solvey(D_in=D_in, x_in=xl[i], E_in=E_in, Z_in=Z_in, sol_guess=10.0)
        
        if xl[i]<(ab_in[0]+ab_in[2]):
            ylQf[i] = np.max([ab_in[3]*np.sqrt(1.0-(xl[i]/ab_in[2])**2),ylQ[i]])
        elif xl[i]>(D_in-(ab_in[0]+ab_in[4])):
            ylQf[i] = np.max([ab_in[5]*np.sqrt(1.0-((D_in-xl[i])/(ab_in[4]))**2),ylQ[i]])
        else:
            ylQf[i] = ylQ[i]
    return ylQ,ylQf
    """

    xl = np.linspace(0.0,D_in,Dsize_in)
    ylQ = np.zeros_like(xl)
    ylQf = np.zeros_like(xl)
    for i in range(0,len(ylQ)):
        ylQ[i] = pint_in.solvey(D_in=D_in, x_in=xl[i], E_in=E_in, Z_in=Z_in, sol_guess=10.0)
        
        if (D_in-(ab_in[0]+ab_in[4])) < xl[i] < (ab_in[0]+ab_in[2]):
            ylQf[i] = np.max([(ab_in[3]+ab_in[1])*np.sqrt(1.0-(xl[i]/(ab_in[2]+ab_in[0]))**2),
                              (ab_in[5]+ab_in[1])*np.sqrt(1.0-((D_in-xl[i])/(ab_in[4]+ab_in[0]))**2),
                              ylQ[i]])
        elif xl[i] < (ab_in[0]+ab_in[2]) and xl[i] < (D_in-(ab_in[0]+ab_in[4])):
            ylQf[i] = np.max([(ab_in[3]+ab_in[1])*np.sqrt(1.0-(xl[i]/(ab_in[2]+ab_in[0]))**2),ylQ[i]])
        elif xl[i] > (D_in-(ab_in[0]+ab_in[4])) and xl[i] > (ab_in[0]+ab_in[2]):
            ylQf[i] = np.max([(ab_in[5]+ab_in[1])*np.sqrt(1.0-((D_in-xl[i])/(ab_in[4]+ab_in[0]))**2),ylQ[i]])
        else:
            ylQf[i] = ylQ[i]
    return xl,ylQ,ylQf

def getEllipsoidAxes(betas_in,rad_in):
    """
    Returns the semimajor (a) and semiminor (b) axes of an ellipsoid, given
    their ratio beta and the radius of the sphere when a=b. The assumption that
    the volume is constant is used, ie r^3 = a*b^2.
    
    :type betas_in: list of floats
    :param betas_in: Ratio between semimajor and semiminor axis: beta = a/b.

    :type rad_in: list of floats
    :param rad_in: Radii of the sphere when semimajor = semiminor axis (a=b).

    :rtype: list of floats
    :returns: List of Semimajor (a) and semiminor (b) axes: [a1,b1,a2,b2,...]
              and list of difference ec = a^2-b^2: [ec1,ec2,...].
    """
    
    ab_out = []
    for r in rad_in:
        ab_out.extend([r,r])
    
    ec_out = [0]*len(betas_in)
    
    for i in range(0,len(betas_in)):
        if not np.allclose(betas_in[i],1):
            ab_out[i*2] = rad_in[i]*betas_in[i]**(2.0/3.0)
            ab_out[i*2+1] = rad_in[i]*betas_in[i]**(-1.0/3.0)
            ec_out[i] = np.sqrt(ab_out[i*2]**2-ab_out[i*2+1]**2)
    
    return ab_out, ec_out


def getCentreOfMass(r_in, m_in):
    """
    Get coordinates of centre of mass for a system, relative to the "lab frame".
    
    :type r_in: list of floats
    :param r_in: Coordinates of the particles: [r1x, r1y, r2x, r2y, r3x, r3y].

    :type m_in: list of floats
    :param m_in: Masses of the particles: [m1, m2, m3].

    :rtype: list of floats
    :returns: x- and y-coordinate of the centre of mass relative to the "lab
              frame".
    """
    x = 0
    y = 0
    
    for i in range(0,len(m_in)):
        x += r_in[2*i]*m_in[i]/np.sum(m_in)
        y += r_in[2*i+1]*m_in[i]/np.sum(m_in)
    
    return x,y


def getUncertanty(A_in):
    """
    Get the uncertainty of a variable (B) due to the given uncertainty of its
    conjugate variable (A). The uncertainty relation: AB <= hbar/2.
    
    :type A_in: float
    :param A_in: Given uncertainty of a variable, whose conjugate variable
                 uncertainty is to be determined.
    
    :rtype: float
    :returns: Uncertainty of a variable due to given uncertainty of its
              conjugate variable.
    """
    
    hbar = 1.0
    B_out = hbar/(2*A_in)
    
    return B_out


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


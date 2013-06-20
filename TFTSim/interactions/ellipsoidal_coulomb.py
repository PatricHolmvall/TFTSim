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
import sympy as sp

class PointParticleCoulomb:
    """
    Uses a Coulomb interaction between particles were each particle is
    modelled as a point, i.e. |F| = k*q1*q2/|r12^2|.

    Contains the equation of motion a = k*q1*q2*r12/(m*|r12^3|) and the Coulomb
    Energy E = -k*q1*q2/|r12|.
    """
    
    def __init__(self, ke2_in = 1.43996518):
        """
        :type ke2: float
        :param ke2: (e^2)/(4*pi*eps0*epsr) in MeV*fm.
                    1.43996518 (44) MeV fm (Sources: M. Aguilar-Benitez, et al., Phys. Lett. 170B (1986) 1
                                                     R.L. Robinson, Science 235 (1987) 633)
        """
        self.ke2 = ke2_in

    def accelerations(self, Z_in, r_in, m_in):
        """
        Calculate the accelerations of all particles due to Coulomb interactions
        with each other through a = k*q1*q2/(m*r12^2).
        
        :type Z_in: list of ints
        :param Z_in: Particle proton numbers [Z1, Z2, Z3].
        
        :type r_in: list of floats
        :param r_in: Coordinates of the particles: [r1x, r1y, r2x, r2y, r3x,
                                                    r3y].
        
        :type m_in: list of floats
        :param m_in: Particle masses [m1, m2, m3].
        
        :rtype: list of floats
        :returns: Particle accelerations [a1x, a1y, a2x, a2y, a3x, a3y].
        """
        
        r12x = r_in[0]-r_in[2]
        r12y = r_in[1]-r_in[3]
        r13x = r_in[0]-r_in[4]
        r13y = r_in[1]-r_in[5]
        r23x = r_in[2]-r_in[4]
        r23y = r_in[3]-r_in[5]
        d12 = np.sqrt((r12x)**2 + (r12y)**2)
        d13 = np.sqrt((r13x)**2 + (r13y)**2)
        d23 = np.sqrt((r23x)**2 + (r23y)**2)
        c12 = self.ke2*(Z_in[0])*(Z_in[1])
        c13 = self.ke2*(Z_in[0])*(Z_in[2])
        c23 = self.ke2*(Z_in[1])*(Z_in[2])
        
        a1x = c12*r12x / (m_in[0] * d12**3) + c13*r13x / (m_in[0] * d13**3)
        a1y = c12*r12y / (m_in[0] * d12**3) + c13*r13y / (m_in[0] * d13**3)
        a2x = -c12*r12x / (m_in[1] * d12**3) + c23*r23x / (m_in[1] * d23**3)
        a2y = -c12*r12y / (m_in[1] * d12**3) + c23*r23y / (m_in[1] * d23**3)
        a3x = -c13*r13x / (m_in[2] * d13**3) - c23*r23x / (m_in[2] * d23**3)
        a3y = -c13*r13y / (m_in[2] * d13**3) - c23*r23y / (m_in[2] * d23**3)
        
        return a1x,a1y,a2x,a2y,a3x,a3y

    def coulombEnergies(self, Z_in, r_in):
        """
        Calculate all the Coulomb energies between three particles.
        
        :type Z_in: list of ints
        :param Z_in: Particle proton numbers [Z1, Z2, Z3].
        
        :type r_in: list of floats
        :param r_in: Coordinates of the particles: [r1x, r1y, r2x, r2y, r3x,
                                                    r3y].
        
        :rtype: list of floats
        :returns: List of Coulomb Energies (in MeV/c^2) between particles
                  [Ec_12, Ec_13, Ec_23].
        """
        
        return [self.ke2*Z_in[0]*Z_in[1]/(np.sqrt((r_in[0]-r_in[2])**2+(r_in[1]-r_in[3])**2)),
                self.ke2*Z_in[0]*Z_in[2]/(np.sqrt((r_in[0]-r_in[4])**2+(r_in[1]-r_in[5])**2)),
                self.ke2*Z_in[1]*Z_in[2]/(np.sqrt((r_in[2]-r_in[4])**2+(r_in[3]-r_in[5])**2))]

    def coulombEnergy(self, Z_in, r_in):
        """
        Calculate the Coulomb energies between two particles.
        
        :type Z_in: list of ints
        :param Z_in: Particle proton numbers [Z1, Z2].
        
        :type r_in: list of floats
        :param r_in: Coordinates of the particles: [r1x, r1y, r2x, r2y].
        
        :rtype: float
        :returns: Coulomb Energies (in MeV/c^2) between the particles.
        """
        
        return self.ke2*Z_in[0]*Z_in[1]/(np.sqrt((r_in[0]-r_in[2])**2+(r_in[1]-r_in[3])**2))

    def solveD(self, xr_in, y_in, E_in, Z_in, r_in, sol_guess):
        """
        Get D for given x, y, Energy, particle Z and particle radii.
        
        :type xr_in: float
        :param xr_in: relative x displacement of ternary particle between heavy and
                     light fission fragment.
        
        :type y_in: float
        :param y_in: y displacement between ternary particle and fission axis.
        
        :type E_in: float
        :param E_in: Energy in MeV.
        
        :type Z_in: list of ints
        :param Z_in: Particle proton numbers [Z1, Z2, Z3].
        
        :type r_in: list of ints
        :param r_in: Particle radii [r1, r2, r3].

        :type sol_guess: float
        :param sol_guess: Initial guess for solution.
        
        :rtype: float
        :returns: D for current equation.
        """
        
        a = (Z_in[0]*Z_in[1])
        b = (Z_in[0]*Z_in[2])
        c = (Z_in[1]*Z_in[2])
        A = r_in[0] + r_in[1] - xr_in*(2*r_in[0] + r_in[1] + r_in[2])
        
        Dval = Symbol('Dval')
        return np.float(nsolve(a/((Dval*xr_in+A)**2+y_in**2)**(0.5) + \
                               b/((Dval*(1-xr_in)-A)**2+y_in**2)**(0.5) + \
                               c/Dval-E_in/self.ke2, Dval, sol_guess))

    def solvey(self, D_in, x_in, E_in, Z_in, sol_guess):
        """
        Get y for given D, x, Energy, particle Z and particle radii.
        
        :type D_in: float
        :param D_in: Distance between heavy and light fission fragment.
        
        :type x_in: float
        :param x_in: x displacement between ternary particle and heavy fragment.
        
        :type E_in: float
        :param E_in: Energy in MeV.
        
        :type Z_in: list of ints
        :param Z_in: Particle proton numbers [Z1, Z2, Z3].

        :type sol_guess: float
        :param sol_guess: Initial guess for solution.
        
        :rtype: float
        :returns: y for current equation.
        """
        #sp.mpmath.mp.dps = 3
        yval = Symbol('yval')
        #A = E_in/self.ke2 - Z_in[1]*Z_in[2]/D_in
        #b = Z_in[0]*Z_in[1]
        #c = Z_in[0]*Z_in[2]
        #B = (x_in**2+yval**2)**(0.5)
        #C = ((D_in-x_in)**2+yval**2)**(0.5)
        
        #return np.float(nsolve(b*C+c*B - A*B*C, yval, sol_guess))
        try:
            ySolution = nsolve(Z_in[0]*Z_in[1]*sp.sqrt((D_in-x_in)**2 + yval**2) + \
                               Z_in[0]*Z_in[2]*sp.sqrt(x_in**2 + yval**2) - \
                               (E_in/self.ke2 - Z_in[1]*Z_in[2]/D_in)*sp.sqrt(x_in**2+yval**2)*sp.sqrt((D_in-x_in)**2+yval**2),
                               yval, sol_guess)
        except ValueError:
            ySolution = 0.0
        return np.float(ySolution)


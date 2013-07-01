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

class EllipsoidalParticleCoulomb:
    """
    Uses a Coulomb interaction between particles were two particles are
    ellipsoids and one is a point particle.

    """
    
    def __init__(self, ec_in, ke2_in = 1.43996518):
        """
        :type ke2: float
        :param ke2: (e^2)/(4*pi*eps0*epsr) in MeV*fm.
                    1.43996518 (44) MeV fm (Sources: M. Aguilar-Benitez, et al., Phys. Lett. 170B (1986) 1
                                                     R.L. Robinson, Science 235 (1987) 633)

        :type ec_in: list of floats
        :param ec_in: ec is the deviation from a sphere to an ellipsoid for a
                      particle: Sqrt(a^2-b^2), where a/b is the semi-major/minor
                      axis respectively. c_in = [c1,c2,c3].
        """
        self.ke2 = ke2_in
        self.ec = ec_in
        self.name = 'ellipsoidal'

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
        q12 = self.ke2*(Z_in[0])*(Z_in[1])
        q13 = self.ke2*(Z_in[0])*(Z_in[2])
        q23 = self.ke2*(Z_in[1])*(Z_in[2])
        
        F12r = q12*(1.0/(d12**2) + \
                    3.0*self.ec[1]**2*(3.0*(r12x/d12)**2-1.0)/(10.0*d12**4))
        F12t = q12*(3.0*self.ec[1]**2*r12x*r12y)/(5.0*d12**6)
        F12x = r12x*F12r/d12 + r12y*F12t
        F12y = r12y*F12r/d12 + r12x*F12t
        
        #print(str(F12r/d12)+'\t'+str(F12t)+'\t'+str(3.0*self.ec[1]**2*(3.0*(r12x/d12)**2-1.0)/(10.0*d12**4))+'\t'+str())
        
        F13r = q13*(1.0/(d13**2) + \
                    3.0*self.ec[2]**2*(3.0*(r13x/d13)**2-1.0)/(10.0*d13**4))
        F13t = q13*(3.0*self.ec[2]**2*r13x*r13y)/(5.0*d13**6)
        F13x = r13x*F13r/d13 + r13y*F13t
        F13y = r13y*F13r/d13 + r13x*F13t
        
        F23r = q23*(1.0/(d23**2) + \
                    3.0*(self.ec[1]**2+self.ec[2]**2)/(5.0*d23**4) + \
                    6.0*(self.ec[1]*self.ec[2])**2/(5.0*d23**6)
                   )
        F23x = r23x*F23r/d23
        F23y = r23y*F23r/d23
        
        a1x = ( F12x + F13x)/m_in[0]
        a1y = ( F12y + F13y)/m_in[0]
        
        a2x = (-F12x + F23x)/m_in[1]
        a2y = (-F12y + F23y)/m_in[1]
        
        a3x = (-F13x - F23x)/m_in[2]
        a3y = (-F13y - F23y)/m_in[2]
        
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
        
        r12x = r_in[0]-r_in[2]
        r12y = r_in[1]-r_in[3]
        r13x = r_in[0]-r_in[4]
        r13y = r_in[1]-r_in[5]
        r23x = r_in[2]-r_in[4]
        r23y = r_in[3]-r_in[5]
        d12 = np.sqrt((r12x)**2 + (r12y)**2)
        d13 = np.sqrt((r13x)**2 + (r13y)**2)
        d23 = np.sqrt((r23x)**2 + (r23y)**2)
        
        return [self.ke2*Z_in[0]*Z_in[1]*(1.0/d12 + \
                                          self.ec[1]**2*(3.0*(r12x/d12)**2-1.0)/(10.0*d12**3)),
                self.ke2*Z_in[0]*Z_in[2]*(1.0/d13 + \
                                          self.ec[2]**2*(3.0*(r13x/d13)**2-1.0)/(10.0*d13**3)),
                self.ke2*Z_in[1]*Z_in[2]*(1.0/d23 + \
                                          (self.ec[1]**2+self.ec[2]**2)/(5.0*d23**3) + \
                                          6.0*self.ec[1]**2*self.ec[2]**2/(25.0*d23**5)
                                          )]
        

    def coulombEnergySpheres(self, Z_in, r_in):
        """
        Calculate the Coulomb energies between two point particles.
        
        :type Z_in: list of ints
        :param Z_in: Particle proton numbers [Z1, Z2].
        
        :type r_in: list of floats
        :param r_in: Coordinates of the particles: [r1x, r1y, r2x, r2y].
        
        :rtype: float
        :returns: Coulomb Energies (in MeV/c^2) between the particles.
        """
        
        return self.ke2*Z_in[0]*Z_in[1]/np.sqrt((r_in[0]-r_in[2])**2+(r_in[1]-r_in[3])**2)

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
        
        #self.ke2 = 1.43996518
        #self.ec = [0, 6.2474511063728828, 5.5901699437494727]
        
        yval = Symbol('yval')
        try:
            ySolution = nsolve(sp.sqrt((D_in-x_in)**2+yval**2)**5*Z_in[1]*(10.0*(x_in**2+yval**2)**2+self.ec[1]**2*(2*x_in**2-yval**2)) + \
                               sp.sqrt(x_in**2+yval**2)**5*Z_in[2]*(10.0*((D_in-x_in)**2+yval**2)**2+self.ec[2]**2*(2*(D_in-x_in)**2-yval**2)) + \
                              -10.0*(sp.sqrt(x_in**2+yval**2)*sp.sqrt((D_in-x_in)**2+yval**2))**5* \
                                    (E_in/self.ke2 - (Z_in[1]*Z_in[2])*(1.0/D_in+(self.ec[1]**2+self.ec[2]**2)/(5.0*D_in**3)))/Z_in[0],
                               yval, sol_guess)
        except ValueError, e:
            #print(str(e))
            ySolution = 0.0
        return np.float(ySolution)


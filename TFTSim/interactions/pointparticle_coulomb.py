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

class PointParticleCoulomb:
    """
    Uses a Coulomb interaction between particles were each particle is
    modelled as a point, i.e. |F| = k*q1*q2/|r12^2|.

    Contains the equation of motion a = k*q1*q2*r12/(m*|r12^3|) and the Coulomb
    Energy E = -k*q1*q2/|r12|.
    """
    
    def __init__(self, ke2_in = 1.439964):
        """
        :type ke2: float
        :param ke2: (e^2)/(4*pi*eps0*eps*10^3) in MeV*fm.
        """
        self.ke2 = ke2_in

    def accelerations(self, Z_in, r_in, m_in):
        """
        Calculate the accelerations of all particles due to Coulomb interactions
        with each other through a = k*q1*q2/(m*r12^2).
        
        :type Z: list of ints
        :param Z: Particle proton numbers [Z_tp, Z_hf, Z_lf]

        :type r: list of floats
        :param r: Particle coordinates [xtp, ytp, xhf, yhf, xlf, ylf]
        
        :type m: float
        :param m: Particle masses [m_tp, m_hf, m_lf]
        
        :rtype: list of floats
        :returns: Particle accelerations [axtp, aytp, axhf, ayhf, axlf, aylf]
        """
        
        c12 = self.ke2*Z_in[0]*Z_in[1]
        c13 = self.ke2*Z_in[0]*Z_in[2]
        c23 = self.ke2*Z_in[1]*Z_in[2]
        r12x = r_in[0]-r_in[2]
        r12y = r_in[1]-r_in[3]
        r13x = r_in[0]-r_in[4]
        r13y = r_in[1]-r_in[5]
        r23x = r_in[2]-r_in[4]
        r23y = r_in[3]-r_in[5]
        d12 = np.sqrt((r_in[0]-r_in[2])**2+(r_in[1]-r_in[3])**2)
        d13 = np.sqrt((r_in[0]-r_in[4])**2+(r_in[1]-r_in[5])**2)
        d23 = np.sqrt((r_in[2]-r_in[4])**2+(r_in[3]-r_in[5])**2)
        
        #print(str(c12)+' '+str(c13)+' '+str(c23)+' '+str(r12x)+' '+str(r12y)+' '+str(r13x)+' '+str(r13y))
        #print(str(r23x)+' '+str(r23y)+' '+str(d12)+' '+str(d13)+' '+str(d23))
        #print('-------------------------------------------------------------')
        
        aout = [ c12*r12x/(m_in[0]*d12**3) + c13*r13x/(m_in[0]*d13**3), # axtp
                 c12*r12y/(m_in[0]*d12**3) + c13*r13y/(m_in[0]*d13**3), # aytp
                -c12*r12x/(m_in[1]*d12**3) + c23*r23x/(m_in[1]*d23**3), # axhf
                -c12*r12y/(m_in[1]*d12**3) + c23*r23y/(m_in[1]*d23**3), # ayhf
                -c13*r13x/(m_in[2]*d13**3) - c23*r23x/(m_in[2]*d23**3), # axlf
                -c13*r13y/(m_in[2]*d13**3) - c23*r23y/(m_in[2]*d23**3)] # aylf
        
        return aout

    def coulombEnergies(self, Z_in, r_in):
        """
        Calculate the Coulomb energies between three particles.
        
        :type Z: list of ints
        :param Z: Particle proton numbers [Z_tp, Z_hf, Z_lf]
        
        :type r: list of floats
        :param r: Coordinates of the particles: [rx1, ry1, rx2, ry2, rx3, ry3]
        
        :rtype: list of floats
        :returns: List of Coulomb Energies (in MeV/c^2) between particles
                  [Ec_12, Ec_13, Ec_23]
        """
        
        return [self.ke2*Z_in[0]*Z_in[1]/(np.sqrt((r_in[0]-r_in[2])**2+(r_in[1]-r_in[3])**2) ),
                self.ke2*Z_in[0]*Z_in[2]/(np.sqrt((r_in[0]-r_in[4])**2+(r_in[1]-r_in[5])**2) ),
                self.ke2*Z_in[1]*Z_in[2]/(np.sqrt((r_in[2]-r_in[4])**2+(r_in[3]-r_in[5])**2) )]


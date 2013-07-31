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
# along with FeynSimul.  If not, see <http://www.gnu.org/licenses/>.

"""
This module defines a single class, :class:`TFTSimArgs` which is used to keep
track of input parameters.
"""

import numpy as np
from TFTSim.tftsim_utils import *

class TFTSimArgs:

    def __init__(self,
                 simulationName,
                 fissionType,
                 coulombInteraction,
                 nuclearInteraction,
                 fissioningParticle,
                 projectileParticle,
                 ternaryParticle,
                 heavyFragment,
                 lightFragment,
                 lostNeutrons,
                 betas,
                 minCoulombEnergy,
                 maxRunsODE,
                 maxTimeODE,
                 neutronEvaporation = False,
                 verbose = True,
                 interruptOnException = True,
                 collisionCheck = False,
                 saveTrajectories = False,
                 saveKineticEnergies = True,
                 useGPU = False,
                 GPU64bitFloat = False):
        """
        Creates an instance of a class that contains different parameters for the run.
        Those parameters that have default values does not necessarily need to be
        defined.

        :type simulationName: string
        :param simulationName: Name of the simulation, to group similar runs together.

        :type fissionType: string
        :param fissionType: Fission type, valid values: LCP, CCT, BF
        
        :type coulombInteraction: :class:`interaction` class instance
        :param coulombInteraction: The Coulomb interaction to use.
        
        :type nuclearInteraction: :class:`interaction` class instance
        :param nuclearInteraction: The nuclear interaction to use.
        
        :type fissioningParticle: :class:`particle` class instance
        :param fissioningParticle: Ternary Particle species.
        
        :type projectileParticle: :class:`particle` class instance
        :param projectileParticle: Projectile Particle species, inducing the
                                   fission. Most common is a thermal neutron.

        :type ternaryParticle: :class:`particle` class instance
        :param ternaryParticle: Ternary Particle species.
        
        :type heavyFragment: :class:`particle` class instance
        :param heavyFragment: Heavy Fragment particle species.
        
        :type lightFragment: :class:`particle` class instance
        :param lightFragment: Light Fragment particle species.

        :type lostNeutrons: int
        :param lostNeutrons: Amount of neutrons lost during the fissioning or
                             due to evaporation.
                             
        :type betas: list of floats
        :param betas: List containing the ratio semimajor/semiminor = a/b axis
                      for each particle: betas = [a1/b1, a2/b2, a3/b3].
        
        :type minCoulombEnergy: float
        :param minCoulombEnergy: End simulation when the Coulomb energy reaches
                                 this percent of initial Coulomb energy.
        
        :type maxRunsODE: int
        :param maxRunsODE: Maximum number of consecutive runs that the ODE
                           solver will do to try to get convergence before
                           giving up. Set to zero to run indefinitely.

        :type maxTime: int
        :param maxTimeODE: Maximum time in seconds that the ODE solver is
                           allowed to run before it is interrupted.
        
        :type neutronEvaporation: boolean
        :param neutronEvaporation: Whether or not to use a statistical
                                   probability to evaporate/eject neutrons from
                                   the fission fragments during mid-flight.
        
        :type verbose: boolean
        :param verbose: Whether or not to output information to the terminal.
        
        :type interruptOnException: boolean
        :param interruptOnException: Whether or not to interrupt when an
                                     exception is raised. Letting the program
                                     continue is a good idea if running many
                                     simulations at once. Then the program will
                                     skip the current run and write the
                                     exception in the status of that run.
        
        :type collisionCheck: boolean
        :param collisionCheck: Whether or not to check for collisions between
                               each ODErun.
        
        :type saveTrajectories: boolean
        :param saveTrajectories: Whether or not to save trajectories to file.
        
        :type saveKineticEnergies: boolean
        :param saveKineticEnergies: Whether or not to save each total kinetic
                                    energy after each ODE run in a shelved file.
                                    
        :type useGPU: boolean
        :param useGPU: Whether or not to use GPU.
        
        :type GPU64bitFloat: boolean
        :param GPU64bitFloat: Whether or not to use double floating point
                              precision on the GPU.
        
        """
        self.simulationName = simulationName
        self.fissionType = fissionType
        self.cint = coulombInteraction
        self.nint = nuclearInteraction
        self.fp = fissioningParticle
        self.pp = projectileParticle
        self.tp = ternaryParticle
        self.hf = heavyFragment
        self.lf = lightFragment
        self.lostNeutrons = lostNeutrons
        self.betas = betas
        self.minEc = minCoulombEnergy
        self.maxRunsODE = maxRunsODE
        self.maxTimeODE = maxTimeODE
        self.neutronEvaporation = neutronEvaporation
        self.verbose = verbose
        self.interruptOnException = interruptOnException
        self.collisionCheck = collisionCheck
        self.saveTrajectories = saveTrajectories
        self.saveKineticEnergies = saveKineticEnergies
        self.useGPU = useGPU
        self.GPU64bitFloat = GPU64bitFloat

        if self.fissionType == 'BF':
            self.mff = []
            self.Z = []
            self.rad = []
        else:
            self.mff = [u2m(self.tp.A)]
            self.Z = [self.tp.Z]
            self.rad = [crudeNuclearRadius(self.tp.A)]

        self.mff.extend([u2m(self.hf.A), u2m(self.lf.A)])
        self.Z.extend([self.hf.Z, self.lf.Z])
        self.rad.extend([crudeNuclearRadius(self.hf.A),
                         crudeNuclearRadius(self.lf.A)])
        self.ab, self.ec = getEllipsoidAxes(self.betas, self.rad)


        # Calculate Q value
        if self.pp != None:
            _mEx_pre_fission = np.sum([self.fp.mEx, self.pp.mEx])
        else:
            _mEx_pre_fission = self.fp.mEx
        
        if self.fissionType == 'BF':
            _mEx_post_fission = np.sum([self.hf.mEx, self.lf.mEx])
        else:
            _mEx_post_fission = np.sum([self.tp.mEx, self.hf.mEx, self.lf.mEx])
        self.Q = getQValue(_mEx_pre_fission, _mEx_post_fission,
                           self.lostNeutrons)



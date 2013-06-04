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

class TFTSimArgs:

    def __init__(self,
                 simulationName,
                 particleInteraction,
                 fissioningParticle,
                 projectileParticle,
                 ternaryParticle,
                 heavyFragment,
                 lightFragment,
                 r,
                 minCoulombEnergy,
                 lostNeutrons,
                 neutronEvaporation = False,
                 verbose = True,
                 interruptOnException = True,
                 saveTrajectories = False):
        """
        Creates an instance of a class that contains different parameters for the run.
        Those parameters that have default values does not necessarily need to be
        defined.

        :type simulationName: string
        :param simulationName: Name of the simulation, to group similar runs together.
        
        :type particleInteraction: :class:`interaction` class instance
        :param particleInteraction: The particle interaction to use.
        
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
        
        :type r: list of floats
        :param r: Initial coordinates for the particles.
        
        :type minCoulombEnergy: float
        :param minCoulombEnergy: End simulation when the Coulomb energy is
                                 below this energy.
        
        :type lostNeutrons: int
        :param lostNeutrons: Amount of neutrons lost during the fissioning or
                             due to evaporation.
        
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
        
        :type saveTrajectories: boolean
        :param saveTrajectories: Whether or not to save trajectories to file.
        """
        self.simulationName = simulationName
        self.pint = particleInteraction
        self.fp = fissioningParticle
        self.pp = projectileParticle
        self.tp = ternaryParticle
        self.hf = heavyFragment
        self.lf = lightFragment
        self.r = r
        self.minEc = minCoulombEnergy
        self.lostNeutrons = lostNeutrons
        self.neutronEvaporation = neutronEvaporation
        self.verbose = verbose
        self.interruptOnException = interruptOnException
        self.saveTrajectories = saveTrajectories
        


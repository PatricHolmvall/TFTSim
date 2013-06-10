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

"""
Demonstrates a simple run on a system of particles , with a pointparticle Coulomb interaction
"""

from __future__ import division
import numpy as np
import pylab as pl
import shelve

from TFTSim.tftsim_args import *
from TFTSim.tftsim import *
from TFTSim.tftsim_utils import *

# Import the desired interaction
from TFTSim.interactions.pointparticle_coulomb import *

# Import the desired particles
from TFTSim.particles.u235 import *
from TFTSim.particles.he4 import *
from TFTSim.particles.te134 import *
from TFTSim.particles.sr96 import *
from TFTSim.particles.n import *

# Import the desired configuration generator (not required if running single simulation)
from TFTSim.generators.generatorOne import *
from TFTSim.generators.generatorTwo import *



plotTrajectories = False
saveTrajectories = False
animateTrajectories = False

sa = TFTSimArgs(simulationName = 'Test',
                fissionType = 'LCP', # LCP, CCT, BF
                particleInteraction = PointParticleCoulomb(),
                fissioningParticle = U235(),
                projectileParticle = N(),
                ternaryParticle = He4(),
                heavyFragment = Te134(),
                lightFragment = Sr96(),
                lostNeutrons = 2,
                minCoulombEnergy = 0.01, # Percent of initial Ec
                maxRunsODE = 100,
                maxTimeODE = 0,
                neutronEvaporation = False,
                verbose = True,
                interruptOnException = False,
                saveTrajectories = saveTrajectories)

# Initial geometry, lenghts given in fm
#h = 2.0
#D1 = 10.0
#D2 = 10.0
#r = [0, h, -D1, 0, D2, 0] # [tpx0, tpy0, hfx0, hfy0, lfx0, lfy0]                
#v = [0.0, 0.0 , 0.0, 0.0, 0.0, 0.0]
# Single run
#sim = SimulateTrajectory(sa, r, v)
#sim.run()

#gen = GeneratorOne(sa, Dmax=15.0, Dinc=0.5, xinc=0.5, yinc=0.5, ymax=15.0, ymin=0.5)
#gen.generate()

gen = GeneratorTwo(sa, sims=3000, DMu=1.5, DSigma=2.0, yMu=1.5, ySigma=2.0, ymin=0.5)
gen.generate()

#shelvedVariables = shelve.open(sim.getFilePath() + 'shelvedVariables.sb')
#for ex in shelvedVariables:
#    print('------------------------------- '+str(ex))
#    print shelvedVariables[ex]['Ec']

if plotTrajectories and not saveTrajectories:
    print("Note that in order to plot trajectories, saveTrajectories needs"
          " to be set to True.")

if plotTrajectories and saveTrajectories:
    sim.plotTrajectories()

if animateTrajectories and not saveTrajectories:
    print("Note that in order to animate trajectories, saveTrajectories needs"
          " to be set to True.")
          
if animateTrajectories and saveTrajectories:
    sim.animateTrajectories()




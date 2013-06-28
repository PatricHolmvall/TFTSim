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
from TFTSim.interactions.pointlike_particle_coulomb import *
from TFTSim.interactions.ellipsoidal_particle_coulomb import *

# Import the desired particles
from TFTSim.particles.u235 import *
from TFTSim.particles.he4 import *
from TFTSim.particles.te134 import *
from TFTSim.particles.sr96 import *
from TFTSim.particles.ce148 import *
from TFTSim.particles.ge84 import *
from TFTSim.particles.n import *

# Import the desired configuration generator (not required if running single simulation)
from TFTSim.generators.generatorOne import *
from TFTSim.generators.generatorTwo import *
from TFTSim.generators.generatorThree import *
from TFTSim.generators.generatorFour import *



plotTrajectories = True
saveTrajectories = True
animateTrajectories = False

FP = U235()
PP = N()
TP = He4()
HF = Te134()
LF = Sr96()
betas = [1,1.5,1.5]
ab1 = getEllipsoidAxes(betas[0],crudeNuclearRadius(TP.A))
ab2 = getEllipsoidAxes(betas[1],crudeNuclearRadius(HF.A))
ab3 = getEllipsoidAxes(betas[2],crudeNuclearRadius(LF.A))
print(ab1)
print(ab2)
print(ab3)
ec = [np.sqrt(ab1[0]**2-ab1[1]**2),np.sqrt(ab2[0]**2-ab2[1]**2),np.sqrt(ab3[0]**2-ab3[1]**2)]
print(ec)

sa = TFTSimArgs(simulationName = 'Test',
                fissionType = 'LCP', # LCP, CCT, BF
                #particleInteraction = PointlikeParticleCoulomb(),
                particleInteraction = EllipsoidalParticleCoulomb(c_in=ec),
                fissioningParticle = FP,
                projectileParticle = PP,
                ternaryParticle = TP,
                heavyFragment = HF,
                lightFragment = LF,
                lostNeutrons = 2,
                betas = betas,
                minCoulombEnergy = 0.01, # Percent of initial Ec
                maxRunsODE = 1000,
                maxTimeODE = 0,
                neutronEvaporation = False,
                verbose = True,
                interruptOnException = False,
                collisionCheck = False,
                saveTrajectories = saveTrajectories,
                saveKineticEnergies = True)

# Initial geometry, lenghts given in fm
D = 20.1
y = 5.0
x = D*0.5
r = [0, y, -x, 0, D-x, 0] # [tpx0, tpy0, hfx0, hfy0, lfx0, lfy0]                
v = [0.0, 0.0 , 0.0, 0.0, 0.0, 0.0]
# Single run
sim = SimulateTrajectory(sa, r, v)
sim.run()

#gen = GeneratorOne(sa, Dmax=15.0, Dinc=0.5, xinc=0.5, yinc=0.5, ymax=15.0, ymin=0.5)
#gen.generate()

#gen = GeneratorTwo(sa, sims=3000, DMu=1.5, DSigma=2.0, yMu=1.5, ySigma=2.0, ymin=0.0)
#gen.generate()

#gen = GeneratorThree(sa, sims=3000, D=18.2, dx=0.0, dy=0.0, dE=0.0)
#gen.generate()

#gen = GeneratorFour(sa, sims=1000, D=18.1, dx=0.0, dy=0.0, dE=0.0,angles=16,radii=5)
#gen.generate()

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




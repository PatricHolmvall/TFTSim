# TFTSim: Ternary Fission Trajectory Simulation in Python.
# Copyright (C) 2013 Patric Holmvall mail: patric.hol {at} gmail {dot} com
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
from TFTSim.interactions.woods_saxon import *
from TFTSim.interactions.proximity_potential import *

# Import the desired particles
from TFTSim.particles.n import *
from TFTSim.particles.he4 import *
from TFTSim.particles.si34 import *
from TFTSim.particles.ca48 import *
from TFTSim.particles.ca50 import *
from TFTSim.particles.ni68 import *
from TFTSim.particles.ni70 import *
from TFTSim.particles.ni72 import *
from TFTSim.particles.ge84 import *
from TFTSim.particles.kr94 import *
from TFTSim.particles.rb95 import *
from TFTSim.particles.sr96 import *
from TFTSim.particles.zr98 import *
from TFTSim.particles.mo102 import *
from TFTSim.particles.cd118 import *
from TFTSim.particles.cd120 import *
from TFTSim.particles.sn132 import *
from TFTSim.particles.te134 import *
from TFTSim.particles.i135 import *
from TFTSim.particles.xe136 import *
from TFTSim.particles.ce148 import *
from TFTSim.particles.u235 import *
from TFTSim.particles.cf252 import *

# Import the desired configuration generator (not required if running single simulation)
from TFTSim.generators.generatorOne import *
from TFTSim.generators.generatorTwo import *
from TFTSim.generators.generatorThree import *
#from TFTSim.generators.generatorFour import *
from TFTSim.generators.generatorFive import *
from TFTSim.generators.generatorSix import *
from TFTSim.generators.binaryGenerator import *
from TFTSim.generators.cctGenerator import *
from TFTSim.generators.regionGenerator import *

"""
FP = U235()
PP = N()
TP = Si34()
#TP = He4()
HF = Sn132()
#HF = Xe136()
#LF = Kr94()
LF = Ni68()
"""
FP = Cf252()
PP = None
TP = Ca50()
HF = Sn132()
LF = Ni70()


betas = [1,1,1]
rad = [crudeNuclearRadius(TP.A), crudeNuclearRadius(HF.A), crudeNuclearRadius(LF.A)]
ab, ec = getEllipsoidAxes(betas, rad)

sa = TFTSimArgs(simulationName = 'Test',
                fissionType = 'CCT', # LCP, CCT, BF
                #particleInteraction = PointlikeParticleCoulomb(),
                coulombInteraction = EllipsoidalParticleCoulomb(ec_in=ec),
                nuclearInteraction = ProximityPotential(),
                fissioningParticle = FP,
                projectileParticle = PP,
                ternaryParticle = TP,
                heavyFragment = HF,
                lightFragment = LF,
                lostNeutrons = 0,
                betas = betas,
                minCoulombEnergy = 0.03, # Fraction of initial Ec
                maxRunsODE = 1000,
                maxTimeODE = 0,
                neutronEvaporation = False,
                verbose = True,
                plotInitialConfigs = False,
                displayGeneratorErrors = True,
                collisionCheck = False,
                saveTrajectories = False,
                trajectorySaveSize = 100000,
                saveKineticEnergies = False,
                useGPU = True,
                GPU64bitFloat = True,
                useNuclearInt = False)

oneSim = False
sim = SimulateTrajectory(sa)
#gen = RegionGenerator(sa=sa, sims=32*448*4, mu_D=18.6, mode="soloD", lineSamples=500)#, saveConfigs = True)
#gen = GeneratorFive(sa=sa, sims=32*448*4)#, saveConfigs = True)#, oldConfigs="results/Test/2013-08-13/11.33.31/initialConfigs.sb")
gen = CCTGenerator(sa=sa, sims=32*448*4, mode="sequential3", deltaDmin=0, deltaDmax=400.0, yMax=0.0, Dcount=8, ycount=448, Ekin0=40.0, IM = Cd120(), saveConfigs = True)#, oldConfigs = "results/Test/2013-09-12/09.44.01/initialConfigs.sb")
sim.run(generator=gen)
#sim.adaptiveRun(generator=gen, adaptiveRuns=1000, stepSize=0.25)

# Initial geometry, lenghts given in fm
"""oneSim = True
D = 30.5593
y = 1.0
x = rad[1] + rad[0] + 0.5
#r = [0.0, 0.0, D, 0.0] # [tpx0, tpy0, hfx0, hfy0, lfx0, lfy0]                
#v = [0.0, 0.0 , 0.0, 0.0]
r = [0, y, -x, 0, D-x, 0] # [tpx0, tpy0, hfx0, hfy0, lfx0, lfy0]                
v = [-0.085, 0.0 , 0.0, 0.0, 0.0, 0.0]
#v = [0.0]*6
# Single run
sim = SimulateTrajectory(sa, r, v)
exceptionCount, outString, ENi = sim.run()
if exceptionCount == 0:
    print(outString)
"""

#oneSim = False
#gen = GeneratorOne(sa, Dmax=15.0, Dinc=0.5, xinc=0.5, yinc=0.5, ymax=15.0, ymin=0.5)
#gen.generate()

#oneSim = False
#gen = GeneratorTwo(sa, sims=3000, DMu=1.5, DSigma=2.0, yMu=1.5, ySigma=2.0, ymin=0.0)
#gen.generate()

#oneSim = False
#gen = GeneratorThree(sa, sims=3000, D=20.1, dx=0.0, dy=0.0, dE=0.0)
#gen.generate()

#oneSim = False
#gen = GeneratorFour(sa, sims=1000, D=18.1, dx=0.0, dy=0.0, dE=0.0,angles=16,radii=5)
#gen.generate()

#oneSim = False
#gen = GeneratorSix(sa, sims=10, Dmax=10, dx=0.5, yMax=0, dy=0, config='max', Ekin0=40)
#gen.generate()

#oneSim = False
#gen = BinaryGenerator(sa, sims=1000, Dmin=18.651, Dmax=30.0, Ekin0=0)
#gen.generate()

#shelvedVariables = shelve.open(sim.getFilePath() + 'shelvedVariables.sb')
#for ex in shelvedVariables:
#    print('------------------------------- '+str(ex))
#    print shelvedVariables[ex]['Ec']



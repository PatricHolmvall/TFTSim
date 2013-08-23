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
Demonstrates some simple data analysis preformed on a number of simulations.
"""

from __future__ import division
import numpy as np
import scipy.interpolate
from scipy.interpolate import griddata
import pylab as pl
import matplotlib as ml
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import shelve
#from TFTSim.interactions.pointparticle_coulomb import *
from TFTSim.tftsim_utils import *
from TFTSim.tftsim_analysis import *

#simulationPath = "results/Test/2013-08-13/15.33.10/"
#simulationPath = "results/Test/2013-08-13/15.54.16/"
#simulationPath = "results/Test/2013-08-13/16.19.33/"
#simulationPath = "results/Test/2013-08-13/16.52.20/"
#simulationPath = "results/Test/2013-08-15/12.01.17/" #CCT - rest uniform
#simulationPath = "results/Test/2013-08-16/10.59.56/" #CCT - rest uniform, rand
#simulationPath = "results/Test/2013-08-16/14.44.30/" #CCT - rest uniform, rand, with Ekin0 < 10
#simulationPath = "results/Test/2013-08-16/15.15.09/" #CCT - rest uniform, rand, with Ekin0 < 20
#simulationPath = "results/Test/2013-08-17/16.18.28/" #CCT - Ni is TP, ekin = 0
#simulationPath = "results/Test/2013-08-17/19.30.54/" #CCT - Ni is TP, ekin < 20
simulationPath = "results/Test/2013-08-23/15.37.52/" #CCT - Ni is TP, ekin < 20 - DOUBLE   

#simulationPath = "results/Test/2013-08-23/12.23.10/" #CCT - Triad setting - GPU 0.001
#simulationPath = "results/Test/2013-08-23/12.26.04/" #CCT - Triad setting - GPU 0.01
#simulationPath = "results/Test/2013-08-23/12.32.03/" #CCT - Triad setting - GPU 0.05
#simulationPath = "results/Test/2013-08-23/12.56.29/" #CCT - Triad setting - GPU 0.001 - DOUBLE
#simulationPath = "results/Test/2013-08-23/13.47.00/" #CCT - Triad setting - GPU 0.01 - DOUBLE
#simulationPath = "results/Test/2013-08-23/13.38.35/" #CCT - Triad setting - GPU 0.05 - DOUBLE
#simulationPath = "results/Test/2013-08-23/12.29.04/" #CCT - Triad setting - CPU 0.001
#simulationPath = "results/Test/2013-08-23/12.30.37/" #CCT - Triad setting - CPU 0.01


simulationPath = "results/Test/2013-08-23/16.53.43/"

da = TFTSimAnalysis(simulationPath = simulationPath, verbose = True)
da.openShelvedVariables()
#da.plotItAll()
da.plotCCT()
#da.plotTrajectories(color='r')

plt.show()


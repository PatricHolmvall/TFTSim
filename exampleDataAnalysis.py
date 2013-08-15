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

simulationPath = "results/Test/2013-08-13/15.33.10/"
#simulationPath = "results/Test/2013-08-13/15.54.16/"
#simulationPath = "results/Test/2013-08-13/16.19.33/"
#simulationPath = "results/Test/2013-08-13/16.52.20/"
simulationPath = "results/Test/2013-08-15/12.01.17/" #CCT

da = TFTSimAnalysis(simulationPath = simulationPath, verbose = True)
da.openShelvedVariables()
#da.plotItAll()
da.plotCCT()
#da.plotTrajectories(color='r')

plt.show()


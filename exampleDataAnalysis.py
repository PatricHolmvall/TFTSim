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

#simulationPath = "results/Test/2013-08-07/13.48.54/shelvedVariables.sb"         # py = py_0
#simulationPath = "results/Test/2013-08-07/14.10.42/shelvedVariables.sb"         # py = sqrt(py_0**2 + pz_0**2)
#simulationPath = "results/Test/2013-08-07/14.25.47/shelvedVariables.sb"         # py = sqrt(py_0**2 + pz_0**2) 60k samples
#simulationPath = "results/Test/2013-08-07/14.30.34/shelvedVariables.sb"         # py = sqrt(py_0**2 + pz_0**2) sigma_D = 2.0
#simulationPath = "results/Test/2013-08-07/14.38.59/shelvedVariables.sb"         # py = sqrt(py_0**2 + pz_0**2) sigma_x,y = 2.0
#simulationPath = "results/Test/2013-08-07/14.38.59/shelvedVariables.sb"         # py = sqrt(py_0**2 + pz_0**2) sigma_x,y = 2.0,1.0 15.26.42
#simulationPath = "results/Test/2013-08-07/15.26.42/shelvedVariables.sb"         # py = sqrt(py_0**2 + pz_0**2) sigma_x,y = 2.0,1.0,center
#simulationPath = "results/Test/2013-08-07/15.35.03/shelvedVariables.sb"         # py = sqrt(py_0**2 + pz_0**2) sigma_x,y = 2.0,2.0,center
simulationPath = "results/Test/2013-08-07/17.24.18/shelvedVariables.sb"         # py = sqrt(py_0**2 + pz_0**2) sigma_x,y = 2.0,2.0,center


#simulationPath = "results/Test/2013-08-07/11.40.31/shelvedVariables.sb"
#simulationPath = "results/Test/2013-08-07/11.52.23/shelvedVariables.sb"


da = TFTSimAnalysis(simulationPath = simulationPath, verbose = True)
da.plotItAll()


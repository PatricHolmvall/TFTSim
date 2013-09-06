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
#simulationPath = "results/Test/2013-08-23/15.37.52/" #CCT, ekin < 20 - DOUBLE   
#simulationPath = "results/Test/2013-08-26/10.47.28/" #CCT, ekin < 20 - DOUBLE   py = py_0
#simulationPath = "results/Test/2013-08-26/13.10.39/" #CCT, ekin < 20 - DOUBLE   py = sqrt

simulationPath = "results/Test/2013-08-28/09.03.13/" #CCT - uncertainty <ekin> = 5, sigma_ekin = 1
simulationPath = "results/Test/2013-08-28/10.04.17/" #CCT - uncertainty <ekin> = 10, sigma_ekin = 1
simulationPath = "results/Test/2013-08-28/13.46.40/" #CCT - uncertainty <ekin> = 10, sigma_ekin = 1, sigma_x = 1, sigma_y = 0.5
simulationPath = "results/Test/2013-08-28/18.16.06/" #CCT - uncertainty <ekin> = 10, sigma_ekin = 1, sigma_x = 2, sigma_y = 0.5
#simulationPath = "results/Test/2013-08-28/17.04.30/" #CCT - uncertainty <ekin> = 5, sigma_ekin = 1, sigma_x = 2, sigma_y = 0.2
simulationPath = "results/Test/2013-08-29/09.48.09/" #CCT - uncertainty <ekin> = 5, sigma_ekin = 1, sigma_x = 2, sigma_y = 0.1
simulationPath = "results/Test/2013-08-29/10.37.29/" #CCT - uncertainty <ekin> = 5, sigma_ekin = 1, sigma_x = 2, sigma_y = 0.1 not-sqrt
simulationPath = "results/Test/2013-08-29/11.34.39/" #CCT - uncertainty <ekin> = 5, sigma_ekin = 1, sigma_x = 2, sigma_y = 0.5 not-sqrt
simulationPath = "results/Test/2013-08-29/12.54.19/" #CCT - uncertainty <ekin> = 5, sigma_ekin = 1, sigma_x = 1, sigma_y = 1 not-sqrt
simulationPath = "results/Test/2013-08-29/13.54.13/" #CCT - uncertainty <ekin> = 5, sigma_ekin = 2, sigma_x = 1, sigma_y = 1 not-sqrt

simulationPath = "results/Test/2013-08-30/10.37.49/" #Good old He4 + Zr98 + Sn132, sqrt
simulationPath = "results/Test/2013-08-30/11.26.22/" #Good old He4 + Zr98 + Sn132, non-sqrt
simulationPath = "results/Test/2013-08-30/12.42.05/" #Good old He4 + Zr98 + Sn132, sqrt, mu_D-3
simulationPath = "results/Test/2013-08-30/14.59.53/" #Good old He4 + Zr98 + Sn132, sqrt, mu_D-3, saddle
simulationPath = "results/Test/2013-08-30/15.50.36/" #Good old He4 + Zr98 + Sn132, sqrt, saddle
simulationPath = "results/Test/2013-08-30/16.15.01/" #Good old He4 + Zr98 + Sn132, sqrt, saddle
simulationPath = "results/Test/2013-08-30/16.41.42/" #Good old He4 + Zr98 + Sn132, sqrt, saddle, mu_y=1
simulationPath = "results/Test/2013-08-30/17.07.16/" #Good old He4 + Zr98 + Sn132, sqrt, mu_y=0, center-2
simulationPath = "results/Test/2013-09-02/10.09.49/" #Good old He4 + Zr98 + Sn132, sqrt, mu_y=0, xr=0.3
simulationPath = "results/Test/2013-09-02/10.48.07/" #Good old He4 + Zr98 + Sn132, sqrt, mu_y=0, xr=0.3, D-2.0
simulationPath = "results/Test/2013-09-02/11.15.04/" #Good old He4 + Zr98 + Sn132, sqrt, mu_y=0, xr=0.3, D-2.0, mu_y=2.0
simulationPath = "results/Test/2013-09-02/12.24.36/" #Good old He4 + Zr98 + Sn132, sqrt, mu_y=0, xr=0.3, D-2.0, mu_y=2.0, sigma_D=0.5
simulationPath = "results/Test/2013-09-02/13.31.35/" #Good old He4 + Zr98 + Sn132, sqrt, D-1.0, mu_y=1.0, sigma_D=0.5
simulationPath = "results/Test/2013-09-02/15.17.17/" #Good old He4 + Sr96 + Te134, sqrt
simulationPath = "results/Test/2013-09-02/15.44.48/" #Good old He4 + Kr94 + Xe136, sqrt
simulationPath = "results/Test/2013-09-02/16.10.39/" #Good old He4 + Kr94 + Xe136, sqrt
simulationPath = "results/Test/2013-09-02/16.29.30/" #Good old He4 + Kr94 + Xe136, sqrt
simulationPath = "results/Test/2013-09-02/16.55.13/" #Good old He4 + Kr94 + Xe136, sqrt
simulationPath = "results/Test/2013-09-03/09.44.45/" #Good old He4 + Kr94 + Xe136, sqrt
simulationPath = "results/Test/2013-09-03/11.33.55/" #Good old He4 + Kr94 + Xe136, sqrt
simulationPath = "results/Test/2013-09-03/13.01.48/" #Good old He4 + Kr94 + Xe136, sqrt
simulationPath = "results/Test/2013-09-03/14.33.21/" #Good old He4 + Kr94 + Xe136, sqrt, v_tp_0 = 0

simulationPath = "results/Test/2013-09-03/15.36.27/" #regionGenerator, D = 19
simulationPath = "results/Test/2013-09-03/16.23.29/" #regionGenerator, D = 18
simulationPath = "results/Test/2013-09-03/16.45.38/" #regionGenerator, D = 18.6

simulationPath = "results/Test/2013-09-04/09.16.35/" #CCT 252Cf->132Sn+48Ca+70Ni+2n, triad
simulationPath = "results/Test/2013-09-04/09.24.28/" #CCT 252Cf->132Sn+48Ca+70Ni+2n, randuniform
simulationPath = "results/Test/2013-09-04/10.12.25/" #CCT 252Cf->132Sn+48Ca+70Ni+2n, randuniform, high Dmax and Ekin0max


simulationPath = "results/Test/2013-09-06/15.08.34/" #CCT 252Cf Sequential, ddmax = 30, TXEmax=50, txestatic=0-TXE, dmin+10
simulationPath = "results/Test/2013-09-06/15.35.06/" #CCT 252Cf Sequential, ddmax = 30, TXEmax=50, txestatic=0-TXE
simulationPath = "results/Test/2013-09-06/16.24.13/" #CCT 252Cf Sequential, ddmax = 50, TXEmax=30, txestatic=0-TXE
#simulationPath = "results/Test/2013-09-06//" #CCT 252Cf Sequential, ddmax = 30, TXEmax=30, txestatic=0


#simulationPath = "results/Test/2013-08-23/12.23.10/" #CCT - Triad setting - GPU 0.001
#simulationPath = "results/Test/2013-08-23/12.26.04/" #CCT - Triad setting - GPU 0.01
#simulationPath = "results/Test/2013-08-23/12.32.03/" #CCT - Triad setting - GPU 0.05
#simulationPath = "results/Test/2013-08-23/12.56.29/" #CCT - Triad setting - GPU 0.001 - DOUBLE
#simulationPath = "results/Test/2013-08-23/13.47.00/" #CCT - Triad setting - GPU 0.01 - DOUBLE
#simulationPath = "results/Test/2013-08-23/13.38.35/" #CCT - Triad setting - GPU 0.05 - DOUBLE
#simulationPath = "results/Test/2013-08-23/12.29.04/" #CCT - Triad setting - CPU 0.001
#simulationPath = "results/Test/2013-08-23/12.30.37/" #CCT - Triad setting - CPU 0.01


da = TFTSimAnalysis(simulationPath = simulationPath, verbose = True)
da.openShelvedVariables()
#da.plotItAll()
da.plotCCT()
#da.plotTrajectories(color='r')

plt.show()


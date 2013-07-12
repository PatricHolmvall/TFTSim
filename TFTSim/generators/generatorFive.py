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
from scipy.stats import truncnorm
from TFTSim.tftsim_utils import *
from TFTSim.tftsim import *
import copy
from time import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.constants import codata
import pylab as pl
        
class GeneratorFive:
    """
    Based on the model proposed by H.M.A. Radi and J.O. Rasmussen et. al. in
    Phys.Rev.C 26(5) 2049 (1982):
    Monte Carlo Studies of alpha-accompanied fission
    
    Short description of the model:
    
    """
    
    def __init__(self, sa, sims,
                 sigma_D = 1.0, sigma_d = 1.04, sigma_x=0.93, sigma_y=1.3,
                 E_TP_inf=15.9, E_TP_sciss=5.0, E_FS_sciss=13.0, E_FS_inf=155.0):
        """
        Initialize and pre-process the simulation data.

        :type sa: :class:`tftsim_args.TFTSimArgs` class
        :param sa: An instance of TFTSimArgs describing what kind of system to
                   simulate.
        
        :type sims: int
        :param sims: Number of simulations to run.
        
        """
        
        # Check for exceptions in input parameters

        self._sa = sa
        self._sims = sims
        self._sigma_D = sigma_D
        self._sigma_d = sigma_d
        self._sigma_x = sigma_x
        self._sigma_y = sigma_y
        
        # Impose uncertainty principle - an uncertaint in x,y will lead to an
        # uncertainty in px,py according to sigma_px = hbar / (2*sigma_x).
        # The final result will be in units MeV/c.
        self._sigma_px = codata.value('Planck constant over 2 pi in eV s')*\
                         1e15/(2.0*self._sigma_x)*1e-6*\
                         codata.value('speed of light in vacuum')
        self._sigma_py = codata.value('Planck constant over 2 pi in eV s')*\
                         1e15/(2.0*self._sigma_y)*1e-6*\
                         codata.value('speed of light in vacuum')
        self._minTol = 0.1
        
        # Solve E_TP_sciss
        
        # Which E to use in the solver
        E_solve = E_TP_inf + E_FS_inf - E_TP_sciss - E_FS_sciss
        
        # Solve D mean value, which we will center our distribution around
        self._dr = 1/(np.sqrt(self._sa.Z[1]/self._sa.Z[2]) + 1.0)
        self._mu_D = self._sa.cint.solveDwhenTPonAxis(xr_in=(1.0-self._dr),
                                                      E_in=E_solve,
                                                      Z_in=self._sa.Z,
                                                      sol_guess=21.0)
        
        # Solve fragment initial kinetic energies
        VH_0 = 0.009 # c
        VL_0 = 0.013 # c
        
        
        
    def generate(self):
        """
        Generate initial configurations.
        """  
        
        dcount = 0
        simTime = time()
        timeStamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
        
        simulationNumber = 0
        
        for i in range(0,self._sims):
            simulationNumber += 1
            # Randomize D
            D = np.random.normal(self._mu_D, self._sigma_D)
            mu_d = D*self._dr
            #d = np.random.norm(mu_d, self._sigma_d)
            
            # Randomize TP placement
            mu_x = D*(1.0-self._dr)
            mu_xyz = [mu_x,0.0,0.0]
            sigma_xyz = [[self._sigma_x, 0.0,           0.0],
                         [0.0,           self._sigma_y, 0.0],
                         [0.0,           0.0,           self._sigma_y]]
            x, y_0, z_0 = np.random.multivariate_normal(mu_xyz, sigma_xyz)
            y = np.sqrt(y_0**2 + z_0**2)
            #x = np.random.norm(d, self._sigma_x)
            #yp = np.random.norm(0.0, self._sigma_y)
            #zp = np.random.norm(0.0, self._sigma_y)
            #y = np.sqrt(yp**2 + zp**2)
            
            # Randomize ternary particle initial kinetic energy
            mu_p = [0.0,0.0,0.0]
            sigma_p = [[self._sigma_px, 0.0,            0.0],
                       [0.0,            self._sigma_py, 0.0],
                       [0.0,            0.0,            self._sigma_py]]
            px, py_0, pz_0 = np.random.multivariate_normal(mu_p, sigma_p)
            ydir = np.sign(py_0)
            py = np.sqrt(py_0**2 + pz_0**2)
            
            # Impose momentum conservation
            
            phx_0 = 0
            plx_0 = 0
            
            if px > 0:
                phx_0 = -px
            elif px < 0:
                plx_0 = px
            
            phly_0 = -py
            
            
            phx = phx_0
            phy = phly_0*0.5
            plx = plx_0
            ply = phly_0*0.5
            
            vx = px / self._sa.mff[0]
            vy = py / self._sa.mff[0]
            vhx = phx / self._sa.mff[1]
            vhy = phy / self._sa.mff[1]
            vlx = plx / self._sa.mff[2]
            vly = ply / self._sa.mff[2]
            
            r = [0,y,-x,0,D-x,0]
            
            Ec0 = self._sa.cint.coulombEnergies(Z_in=self._sa.Z,r_in=r,fissionType_in=self._sa.fissionType)
            Eav = self._sa.Q - np.sum(Ec0)
            
            # Randomize a kinetic energy for the ternary particle
            #Ekin_tot = np.random.uniform(0.0,Eav*0.9)
            
            # Randomize initial direction for the initial momentum of tp
            """
            p2 = Ekin_tot * 2 * self._sa.mff[0]
            px2 = np.random.uniform(0.0,p2)
            py2 = p2 - px2
            xdir = np.random.randint(2)*2 - 1
            ydir = np.random.randint(2)*2 - 1
            """
            
            #v = [xdir*np.sqrt(px2)/self._sa.mff[0],np.sqrt(py2)/self._sa.mff[0],0.0,0.0,0.0,0.0]
            v = [vx,vy,vhx,vhy,vlx,vly]
            
            #v = [0.0]*6
            
            sim = SimulateTrajectory(sa=self._sa, r_in=r, v_in=v)
            e, outString = sim.run(simulationNumber=simulationNumber, timeStamp=timeStamp)
            #sim.plotTrajectories()
            if e == 0:
                print("S: "+str(simulationNumber)+"/~"+str(self._sims)+"\t"+str(r)+"\t"+outString)
        print("Total simulation time: "+str(time()-simTime)+"sec")
        #pl.show()


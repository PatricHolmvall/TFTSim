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
        
class GeneratorFive:
    """
    Based on the model proposed by H.M.A. Radi and J.O. Rasmussen et. al. in
    Phys.Rev.C 26(5) 2049 (1982):
    Monte Carlo Studies of alpha-accompanied fission
    
    Short description of the model:
    
    """
    
    def __init__(self, sa, sims, sigma_D = 1.0, sigma_d = 1.04, sigma_x=0.93, sigma_y=1.3):
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
        # The final result will be in units eV/(m/s).
        self._sigma_px = codata.value('Planck constant over 2 pi in eV s')*\
                         1e15/(2*self._sigma_x)#*\
        #                 codata.value('speed of light in vacuum')
        self._sigma_py = codata.value('Planck constant over 2 pi in eV s')*\
                         1e15/(2*self._sigma_y)#*\
        #                 codata.value('speed of light in vacuum')
        
        self._minTol = 0.1
        
        DELTA_E_236_U_sciss = 4 # MeV
        EKT_236U_sciss = 13 # MeV
        EKT_236U_inf = 155.5 # +- 0.8 MeV
        E_ALPHA_inf = 15.9 # +- 0.2 MeV
        E_ALPHA_sciss = 5 # MeV
        
        # Solve E_TP_sciss
        E_TP_sciss = E_ALPHA_sciss
        E_TP_inf = E_ALPHA_inf
        EKT_FS_sciss = EKT_23U_sciss
        EKT_FS_inf = EKT_23U_inf
        
        
        E_solve = E_TP_inf + EKT_FS_inf - E_TP_sciss - EKT_FS_sciss
        # Solve D
        self._dr = np.sqrt(self._Z[1]/self._Z[2]) + 1.0
        mu_D = self._sa.pint.solveDwhenTPonAxis(xr_in=(1.0-dr),
                                                E_in=E_solve,
                                                Z_in=self._sa.Z,
                                                sol_guess=18.0)
        
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
            # Randomize D
            D = np.random.norm(self._mu_D, self._sigma_D)
            
            # Randomize d
            mu_d = D/dr
            d = np.random.norm(mu_d, self._sigma_d)
            
            # Randomize ternary particle initial kinetic energy
            
            
            r = [0,y,d-D,0,d,0]
            
            Ec0 = self._sa.cint.coulombEnergies(Z_in=self._sa.Z, r_in=r,fissionType_in=self._sa.fissionType)
            Eav = self._sa.Q - np.sum(Ec0)
            
            v1 = 0
            v2 = 0
            
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
            v = [v1,v2,0,0,0,0]
            
            sim = SimulateTrajectory(sa=self._sa, r_in=r, v_in=v)
            e, outString = sim.run(simulationNumber=simulationNumber, timeStamp=timeStamp)
            
            if e == 0:
                print("S: "+str(simulationNumber)+"/~"+str(self._sims)+"\t"+str(r)+"\t"+outString)
        print("Total simulation time: "+str(time()-simTime)+"sec")


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

from __future__ import division
import numpy as np
from scipy.stats import truncnorm
from TFTSim.tftsim_utils import *
from TFTSim.tftsim import *
import copy
import os
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
                 sigma_D = 1.0, sigma_d = 1.04, sigma_x=1.0, mu_d="saddle",
                 ETP_inf=15.8, ETP_sciss=3.1, EKT_sciss=13.0, EKT_inf=155.0):
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
        self._sigma_y = 1.0
        self._mu_d = mu_d
        
        self._EKT_sciss = EKT_sciss
        self._ETP_sciss = ETP_sciss
        
        # Impose uncertainty principle - an uncertaint in x,y will lead to an
        # uncertainty in px,py according to sigma_px = hbar / (2*sigma_x).
        # The final result will be in units MeV/c.
        self._sigma_px = codata.value('Planck constant over 2 pi in eV s')*\
                         1e15*1e-6/(2.0*self._sigma_x)*\
                         codata.value('speed of light in vacuum')
        self._sigma_py = codata.value('Planck constant over 2 pi in eV s')*\
                         1e15*1e-6/(2.0*self._sigma_y)*\
                         codata.value('speed of light in vacuum')
        self._minTol = 0.1
        
        # Solve E_TP_sciss
        
        # Which E to use in the solver
        E_solve = ETP_inf + EKT_inf - ETP_sciss - EKT_sciss
        
        # Solve D mean value, which we will center our distribution around
        if mu_d == "center":
            self._dr = 0.5
        else:
            self._dr = 1/(np.sqrt(self._sa.Z[1]/self._sa.Z[2]) + 1.0)
        self._mu_D = self._sa.cint.solveDwhenTPonAxis(xr_in=(1.0-self._dr),
                                                      E_in=E_solve,
                                                      Z_in=self._sa.Z,
                                                      sol_guess=21.0)
        #self._mu_D = 30.0
        
    def generate(self, onlyCheckInitialConfigs=False,
                 saveConfigs=False, oldConfigs=None):
        """
        Generate initial configurations.
        """  
        
        genTime = time()
        
        sim = SimulateTrajectory(sa=self._sa)
        
        if oldConfigs == None or not os.path.isfile(oldConfigs):
            if onlyCheckInitialConfigs:
                vx_plot = [0]*self._sims
                vy_plot = [0]*self._sims
                vy2_plot = [0]*self._sims
                
                ekin_plot = [0]*self._sims
                ekinh_plot = [0]*self._sims
                ekinl_plot = [0]*self._sims
                Ds = [0]*self._sims
                costheta = [0]*self._sims
                sintheta = [0]*self._sims
                Erot = [0]*self._sims
            
            
            rs = np.zeros([self._sims,6])
            vs = np.zeros([self._sims,6])
            
            
            print('Generating '+str(self._sims)+' initial configurations ...')
            
            i = 0
            while i < self._sims:
                Eav = -1
                while(Eav < 1):
                    # Randomize D
                    D = np.random.normal(self._mu_D, self._sigma_D)
                    
                    # Randomize TP placement
                    mu_x = D*(1.0-self._dr)
                    #mu_xyz = [mu_x,0.0,0.0]
                    #sigma_xyz = [[self._sigma_x, 0.0,           0.0],
                    #             [0.0,           self._sigma_y, 0.0],
                    #             [0.0,           0.0,           self._sigma_y]]
                    #x, y_0, z_0 = np.random.multivariate_normal(mu_xyz, sigma_xyz)
                    x = np.random.normal(mu_x, self._sigma_x)
                    y_0 = np.random.normal(0.0, self._sigma_y)
                    z_0 = np.random.normal(0.0, self._sigma_y)
                    #y = np.sqrt(y_0**2 + z_0**2)
                    y = y_0
                    
                    # Start positions
                    r = [0,y,-x,0,D-x,0]
                    
                    Ec0 = self._sa.cint.coulombEnergies(Z_in=self._sa.Z,r_in=r,fissionType_in=self._sa.fissionType)
                    Eav = self._sa.Q - np.sum(Ec0)
                
                # Get Center of Mass coordinates
                xcm,ycm = getCentreOfMass(r_in=r, m_in=self._sa.mff)
                rcm = [-xcm, y-ycm, -x-xcm, -ycm, (D-x)-xcm, -ycm]
                
                eav2 = -1
                while eav2 < 0:
                    # Randomize ternary particle initial kinetic energy
                    mu_p = [0.0,0.0,0.0]
                    sigma_p = [[self._sigma_px, 0.0,            0.0],
                               [0.0,            self._sigma_py, 0.0],
                               [0.0,            0.0,            self._sigma_py]]
                    #px, py_0, pz_0 = np.random.multivariate_normal(mu_p, sigma_p)
                    px = np.random.normal(0.0, self._sigma_px)
                    py_0 = np.random.normal(0.0, self._sigma_py)
                    pz_0 = np.random.normal(0.0, self._sigma_py)
                    ydir = np.sign(py_0)
                    #py = np.sqrt(py_0**2 + pz_0**2)
                    # Project p_yz upon r_y
                    py2 = (py_0*y_0 + pz_0*z_0)/(y)
                    py = py_0
                    #py = py2
                    Ekin_tp = 0.5 / self._sa.mff[0] * (px**2 + py**2)
                    eav2 = Eav - Ekin_tp

                vtpx = px/self._sa.mff[0]
                vtpy = py/self._sa.mff[0]
                
                
                # Project onto position vector
                costheta = (py_0*y_0 + pz_0*z_0)/(np.sqrt(py_0**2+pz_0**2)*np.sqrt(y_0**2+z_0**2))
                sintheta = np.sqrt(1.0 - costheta**2)
                
                # Randomize kinetic energy of fission fragments
                Eff = np.random.normal(self._EKT_sciss, 1.0)
                
                # Get fission fragment momenta due to conservation of linear and
                # angular momenta of the entire system.
                A = rcm[0]*py - rcm[1]*px
                B = (A + rcm[3]*px - rcm[2]*py)/(rcm[4]-rcm[2]) 
                C = 2*Eff*self._sa.mff[1]*self._sa.mff[2]
                
                plfy = -B
                phfy = -plfy - py
                
                # Polynomial for p_lf_y
                polynomial = [self._sa.mff[1]+self._sa.mff[2],
                              2*self._sa.mff[2]*px,
                              -C + self._sa.mff[2]*(px**2 + phfy**2) + self._sa.mff[1]*plfy**2]
                sols = np.roots(polynomial)

                plfx = max(sols)
                phfx = -plfx - px
                # Check that real solutions exist
                if np.iscomplex(plfx):
                    print(Ekin_tp)
                    raise ValueError('Complex root: '+str(sols)+' (Eav='+str(Eav-Ekin_tp-Eff)+')')
                
                # Verify that total lin. and ang. mom. is zero
                ptotx = px + phfx + plfx
                ptoty = py + phfy + plfy
                angmom = -y*px - x*phfy + (D-x)*plfy
                if not np.allclose(ptotx,0.0):
                    raise ValueError(str(i)+"Linear mom. not conserved: ptotx = "+str(ptotx)+"\tp: ["+str(px)+","+str(phfx)+","+str(plfx)+"]")
                if not np.allclose(ptoty,0.0):
                    raise ValueError(str(i)+"Linear mom. not conserved: ptoty = "+str(ptoty)+"\tp: ["+str(py)+","+str(phfy)+","+str(plfy)+"]")
                if not np.allclose(angmom,0.0):
                    raise ValueError(str(i)+"Angular mom. not conserved: angmom = "+str(angmom)+"\tp: ["+str(-y*px)+","+str(-x*phfy)+","+str((D-x)*plfy)+"]")
                
                # Calculate speeds of fission fragments
                vhfx = phfx / self._sa.mff[1]
                vhfy = phfy / self._sa.mff[1]
                vlfx = plfx / self._sa.mff[2]
                vlfy = plfy / self._sa.mff[2]
                
                # Initial velocities            
                v = [vtpx,vtpy,vhfx,vhfy,vlfx,vlfy]
                
                # Check that initial conditions are valid
                initErrors = sim.checkConfiguration(r_in=r, v_in=v, TXE_in=0.0)
                #sim = SimulateTrajectory(sa=self._sa, r_in=r, v_in=v, TXE=0.0)
                #initErrors = sim.getExceptionCount()
                
                # Store initial conditions if they are good
                if initErrors == 0 and (Eff + Ekin_tp + np.sum(Ec0)) < self._sa.Q:

                    rs[i] = r
                    vs[i] = v
                    
                    if onlyCheckInitialConfigs:
                        Ds[i] = D
                        costhetas[i] = costheta
                        sinthetas[i] = sintheta
                        vx_plot[i] = vtpx/0.011
                        vy_plot[i] = vtpy/0.011
                        vy2_plot[i] = py2/self._sa.mff[0]/0.011
                        ekin_plot[i] = Ekin_tp
                        ekinh_plot[i] = 0.5*(phfx**2 + phfy**2)/self._sa.mff[1]
                        ekinl_plot[i] = 0.5*(plfx**2 + plfy**2)/self._sa.mff[2]


                    i += 1
                    if i%10000 == 0:
                        print(str(i)+' of '+str( self._sims)+' initial conditions generated')
            # end of simulation while loop
            
            print('Generating '+str(self._sims)+' initial configurations took '+str(time()-genTime)+' s.')
            
            # Save initial configurations
            if saveConfigs and oldConfigs == None:
                config_timeStamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S/")
                config_filePath = "data/configs/generatorFive/" + config_timeStamp
                
                # Create results folder if it doesn't exist
                if not os.path.exists(config_filePath):
                    os.makedirs(config_filePath)
                # Store static variables in a shelved file format
                s = shelve.open(config_filePath + 'initialConfigs.sb')
                try:
                    s["0"] = { 'r0': rs, 'v0': vs, 'sims': self._sims}
                finally:
                    s.close()
        else:
            sv = shelve.open(oldConfigs)
            for row in sv:
                rs = sv[row]['r0']
                vs = sv[row]['v0']
                self._sims = sv[row]['sims']
                print sv[row]['sims']
        
        simTime = time()
        print('Running simulations ...')
        
        timeStamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
        
        if self._sa.useGPU:
            sim.runGPU(simulations=self._sims, rs_in=rs, vs_in=vs)
        else:
            for i in range(0,self._sims):
                print("S: "+str(i+1)+"/~"+str(self._sims)+"\t"),
                e, outString = sim.runCPU(r_in = list(rs[i]),
                                          v_in = list(vs[i]),
                                          TXE_in = 0.0,
                                          simulationNumber = (i+1),
                                          timeStamp = timeStamp)
                print(outString)
        
        #plt.show()
        if onlyCheckInitialConfigs:
            fig = plt.figure(0)
            ax = fig.add_subplot(111)
            nx, binsx, patches = ax.hist(Ds, bins=50)
            bincentersx = 0.5*(binsx[1:]+binsx[:-1])
            # add a 'best fit' line for the normal PDF
            #y = mlab.normpdf( bincenters)
            l = ax.plot(bincentersx, nx, 'r--', linewidth=4,label=str('<D> = '+str(np.mean(Ds))))
            ax.set_title('D distribution')
            ax.set_xlabel('D [fm]')
            ax.set_ylabel('Counts')
            ax.legend()
            
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            nx, binsx, patches = ax.hist(vx_plot, bins=50)
            bincentersx = 0.5*(binsx[1:]+binsx[:-1])
            # add a 'best fit' line for the normal PDF
            #y = mlab.normpdf( bincenters)
            l = ax.plot(bincentersx, nx, 'r--', linewidth=4,label='vx')
            ax.set_title('Initial velocity distribution')
            ax.set_xlabel('Vx (in units of V0=0.011c)')
            ax.set_ylabel('Counts')

            fig = plt.figure(2)
            ax = fig.add_subplot(111)
            ny, binsy, patches = ax.hist(vy_plot, bins=50)
            bincentersy = 0.5*(binsy[1:]+binsy[:-1])
            l = ax.plot(bincentersy, ny, 'r--', linewidth=4,label='vy')
            ax.set_title('Initial velocity distribution')
            ax.set_xlabel('Vy (in units of V0=0.011c)')
            ax.set_ylabel('Counts')
            #ax.set_xlim([0,14.001])
            ax.legend()

            fig = plt.figure(3)
            ax2 = fig.add_subplot(111)
            n, bins, patches = ax2.hist(ekin_plot, bins=50)
            bincenters = 0.5*(bins[1:]+bins[:-1])
            # add a 'best fit' line for the normal PDF
            #y = mlab.normpdf( bincenters)
            l = ax2.plot(bincenters, n, 'r--', linewidth=4,label=str("Ea mean: %1.1f MeV" % np.mean(ekin_plot)))
            ax2.set_title('Initial alpha kinetic energy')
            ax2.set_xlabel('Ekin (MeV)')
            ax2.set_ylabel('Counts')
            ax2.legend()

            fig = plt.figure(4)
            ax2 = fig.add_subplot(111)
            n, bins, patches = ax2.hist(ekinh_plot, bins=50)
            bincenters = 0.5*(bins[1:]+bins[:-1])
            # add a 'best fit' line for the normal PDF
            #y = mlab.normpdf( bincenters)
            l = ax2.plot(bincenters, n, 'r--', linewidth=4,label=str("Ehf mean: %1.1f MeV" % np.mean(ekinh_plot)))
            ax2.set_title('Initial heavy kinetic energy')
            ax2.set_xlabel('Ekin (MeV)')
            ax2.set_ylabel('Counts')
            ax2.legend()

            fig = plt.figure(5)
            ax2 = fig.add_subplot(111)
            n, bins, patches = ax2.hist(ekinl_plot, bins=50)
            bincenters = 0.5*(bins[1:]+bins[:-1])
            # add a 'best fit' line for the normal PDF
            #y = mlab.normpdf( bincenters)
            l = ax2.plot(bincenters, n, 'r--', linewidth=4,label=str("Elf mean: %1.1f MeV" % np.mean(ekinl_plot)))
            ax2.set_title('Initial light kinetic energy')
            ax2.set_xlabel('Ekin (MeV)')
            ax2.set_ylabel('Counts')
            ax2.legend()
            
            #fig = plt.figure(6)
            #ax2 = fig.add_subplot(111)
            #n, bins, patches = ax2.hist(costheta, bins=50)
            #bincenters = 0.5*(bins[1:]+bins[:-1])
            #l = ax2.plot(bincenters, n, 'r--', linewidth=4)
            #ax2.set_title('Cos theta')
            #ax2.set_xlabel('Cos theta')
            #ax2.set_ylabel('Counts')
            #ax2.legend()
            print("Ea mean: "+str(np.mean(ekin_plot)))
            print("D mean: "+str(np.mean(Ds)))
        
        print("Simulation run time: "+str(time()-simTime)+"sec")
        plt.show()


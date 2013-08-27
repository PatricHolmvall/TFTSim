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
#from guppy import hpy
from time import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.constants import codata
import pylab as pl
from sympy import Symbol
from sympy.solvers import nsolve
import sympy as sp
        
class CCTGenerator:
    """
    Generator for collinear configurations.
    
    """
    
    def __init__(self, sa, sims, mode, deltaDmin, deltaDmax, yMax, Dcount, ycount,
                 sigma_x = 1.0, sigma_y = 1.0,
                 Ekin0 = 0, saveConfigs = False, oldConfigs = None):# Dmax, dx=0.5, yMax=0, dy=0, config='', Ekin0=0):
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
        if mode not in ["restUniform","randUniform","uniform","uncertainty","triad"]:
            raise ValueError("Selected mode doesn't exist. Valid modes: "+str(["restUniform","uniform","uncertainty","triad"]))
        self._mode = mode
        self._saveConfigs = saveConfigs
        self._oldConfigs = oldConfigs
        if deltaDmin >= deltaDmax:
            raise ValueError("deltaDmin needs to be smaller than deltaDmax")
        self._deltaDmin = deltaDmin
        self._deltaDmax = deltaDmax
        self._yMax = yMax
        self._Dcount = Dcount
        self._ycount = ycount
        self._sigma_x = sigma_x
        self._sigma_y = sigma_y
        self._sigma_px = codata.value('Planck constant over 2 pi in eV s')*\
                         1e15*1e-6/(2.0*self._sigma_x)*\
                         codata.value('speed of light in vacuum')
        self._sigma_py = codata.value('Planck constant over 2 pi in eV s')*\
                         1e15*1e-6/(2.0*self._sigma_y)*\
                         codata.value('speed of light in vacuum')
        self._Ekin0 = Ekin0
        self._xcount = int(self._sims/(self._Dcount*self._ycount))
        #self._Dmax = Dmax
        #self._dx = dx
        #self._yMax = yMax
        #self._dy = dy
        #self._config = config
        #self._Ekin0 = Ekin0
        
        
        self._minTol = 0.0001

        self._dcontact = self._sa.ab[0] + self._sa.ab[4]
        self._xcontact = self._sa.ab[0] + self._sa.ab[2]
        self._dsaddle = 1.0/(np.sqrt(self._sa.Z[1]/self._sa.Z[2])+1.0)
        self._xsaddle = 1.0/(np.sqrt(self._sa.Z[2]/self._sa.Z[1])+1.0)
        
        _setDminDmax(self, Ekin0_in=self._Ekin0)

    def generate(self, filePath=None, plotInitialConfigs=False, verbose=False):
        """
        Generate initial configurations.
        """
        if filePath == None:
            timeStamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
            filePath = "results/noname/" + str(timeStamp) + "/"
        
        # SimulatTrajectory object used to call checkConfig to verify that an
        # initial configuration is valid
        sim = SimulateTrajectory(sa=self._sa)
        
        if self._oldConfigs == None or not os.path.isfile(self._oldConfigs):
            if plotInitialConfigs:
                x_plot = [0]*self._sims
                y_plot = [0]*self._sims
                Ds = [0]*self._sims
                Ecs = [0]*self._sims
                        
            rs = np.zeros([self._sims,6])
            vs = np.zeros([self._sims,6])
            TXE = np.zeros(self._sims)


            D_linspace = np.linspace(self._Dmin, self._Dmax, self._Dcount)
            y_linspace = np.linspace(0.0, self._yMax, self._ycount)
            
            badOnes = 0
            s = 0
            
            if self._mode == "randUniform":
                while s < self._sims:
                    initErrors = 0
                    
                    # Randomize kinetic energy
                    self._Ekin0 = np.random.random() * self._Ekin0
                    _setDminDmax(self, Ekin0_in=self._Ekin0)
                    
                    # Randomize D
                    D = np.random.random() * (self._Dmax - self._Dmin) + self._Dmin
                    # Randomize y
                    y = np.random.random() * self._yMax
                    
                    # Calculate x-range
                    A = ((self._sa.Q-self._minTol-self._Ekin0)/1.43996518-self._sa.Z[1]*self._sa.Z[2]/D)/self._sa.Z[0]
                    polyn = [A,(self._sa.Z[2]-self._sa.Z[1]-A*D),D*self._sa.Z[1]]
                    sols = np.roots(polyn)
                    #print(str(D[i])+'\t'),
                    #print(sols),
                    
                    # Check 2 sols
                    if len(sols) != 2:
                        raise Exception('Wrong amount of solutions: '+str(len(sols)))

                    # Calculate xmin and xmax
                    xmin = max(sols[1],self._sa.ab[0]+self._sa.ab[2]+self._minTol)
                    xmax = min(sols[0],D-(self._sa.ab[0]+self._sa.ab[4]+self._minTol))
                    
                    # Randomize x
                    x = np.random.random() * (xmax - xmin) + xmin
                    r = [0,y,-x,0,D-x,0]
                            
                    Ec0 = self._sa.cint.coulombEnergies(Z_in=self._sa.Z,r_in=r,fissionType_in=self._sa.fissionType)
                    Eav = self._sa.Q - np.sum(Ec0)
                    # Get Center of Mass coordinates
                    xcm,ycm = getCentreOfMass(r_in=r, m_in=self._sa.mff)
                    rcm = [-xcm, y-ycm, -x-xcm, -ycm, (D-x)-xcm, -ycm]
                    
                    # Get initial momenta
                    if not np.allclose(0, self._Ekin0):
                        phfx0 = -np.sqrt(2.0*self._Ekin0/(1.0/self._sa.mff[1] + 1.0/self._sa.mff[2]))
                        plfx0 =  np.sqrt(2.0*self._Ekin0/(1.0/self._sa.mff[1] + 1.0/self._sa.mff[2]))
                        
                        if x < 0.5*D:
                            plfx = plfx0
                            polyn = [(self._sa.mff[1]/self._sa.mff[0] + 1.0),
                                     -2.0*phfx0*self._sa.mff[1]/self._sa.mff[0],
                                     (self._sa.mff[1]/self._sa.mff[0] - 1.0)*phfx0**2]
                            sols = np.roots(polyn)
                            #print(str(D[i])+'\t'),
                            #print(sols),
                            
                            # Check 2 sols
                            if len(sols) != 2:
                                raise Exception('Wrong amount of solutions: '+str(len(sols)))
                            phfx = min(sols)
                            
                            ptpx = phfx0 - phfx
                        else:
                            phfx = phfx0
                            polyn = [(self._sa.mff[2]/self._sa.mff[0] + 1.0),
                                     -2.0*plfx0*self._sa.mff[2]/self._sa.mff[0],
                                     (self._sa.mff[2]/self._sa.mff[0] - 1.0)*plfx0**2]
                            sols = np.roots(polyn)
                            #print(str(D[i])+'\t'),
                            #print(sols),
                            
                            # Check 2 sols
                            if len(sols) != 2:
                                raise Exception('Wrong amount of solutions: '+str(len(sols)))
                            plfx = min(sols)
                            
                            ptpx = plfx0 - plfx
                        
                        ptpy = 0
                        phfy = 0
                        plfy = 0
                    else:    
                        ptpx = 0
                        ptpy = 0
                        phfx = 0
                        phfy = 0
                        plfx = 0
                        plfy = 0
                    
                    # Initial velocities            
                    v = [ptpx / self._sa.mff[0],
                         ptpy / self._sa.mff[0],
                         phfx / self._sa.mff[1],
                         phfy / self._sa.mff[1],
                         plfx / self._sa.mff[2],
                         plfy / self._sa.mff[2]]
                        
                    if not np.allclose(0.0, (ptpx + phfx + plfx)):
                        print ptpx
                        print phfx
                        print plfx
                        raise Exception("Non-zero momenta.")
                        
                    if not np.allclose(self._Ekin0, (v[0]**2*0.5*self._sa.mff[0] +\
                                                     v[2]**2*0.5*self._sa.mff[1] +\
                                                     v[4]**2*0.5*self._sa.mff[2])):
                        print s
                        print (str(v[0]**2*0.5*self._sa.mff[0])+"+"+\
                               str(v[2]**2*0.5*self._sa.mff[1])+"+"+\
                               str(v[4]**2*0.5*self._sa.mff[2])+"="+
                               str(v[0]**2*0.5*self._sa.mff[0] +\
                                   v[2]**2*0.5*self._sa.mff[1] +\
                                   v[4]**2*0.5*self._sa.mff[2])+"!="+\
                               str(self._Ekin0))
                        raise Exception("Wrong kinetic energy.")
            
                    # Check that initial conditions are valid
                    initErrors += sim.checkConfiguration(r_in=r, v_in=v, TXE_in=0.0)
                    #sim = SimulateTrajectory(sa=self._sa, r_in=r, v_in=v, TXE=0.0)
                    #initErrors = sim.getExceptionCount()
            
                    if (np.sum(Ec0) + self._Ekin0) > self._sa.Q:
                        initErrors += 1
                    
                    # Store initial conditions if they are good
                    if initErrors > 0:
                        badOnes += 1
                    else:
                        rs[s] = r
                        vs[s] = v
                        
                        if plotInitialConfigs:
                            Ds[s] = D
                            x_plot[s] = x
                            y_plot[s] = y
                            Ecs[s] = np.sum(Ec0)
                        
                        s+=1
                        
                        if s%5000 == 0 and s > 0 and verbose:
                            print(str(s)+" of "+str(self._sims)+" initial "
                                  "conditions generated.")
                # end of for loop
            elif self._mode == "restUniform":
                for i in range(0,self._Dcount):
                    D = D_linspace[i]
                    
                    for j in range(0,self._ycount):
                        y = y_linspace[j]
                        
                        # Calculate x-range
                        A = ((self._sa.Q-self._minTol-self._Ekin0)/1.43996518-self._sa.Z[1]*self._sa.Z[2]/D)/self._sa.Z[0]
                        p = [A,(self._sa.Z[2]-self._sa.Z[1]-A*D),D*self._sa.Z[1]]
                        sols = np.roots(p)
                        #print(str(D[i])+'\t'),
                        #print(sols),
                        
                        # Check 2 sols
                        if len(sols) != 2:
                            raise Exception('Wrong amount of solutions: '+str(len(sols)))

                        xmin = max(sols[1],self._sa.ab[0]+self._sa.ab[2]+self._minTol)
                        xmax = min(sols[0],D-(self._sa.ab[0]+self._sa.ab[4]+self._minTol))
                        
                        
                        x_linspace = np.linspace(xmin, xmax, self._xcount)
                        
                        for k in range(0,self._xcount):
                            initErrors = 0
                            x = x_linspace[k]
                        
                            # Start positions
                            r = [0,y,-x,0,D-x,0]
                            
                            #print(r)
                    
                            Ec0 = self._sa.cint.coulombEnergies(Z_in=self._sa.Z,r_in=r,fissionType_in=self._sa.fissionType)
                            Eav = self._sa.Q - np.sum(Ec0)
                            # Get Center of Mass coordinates
                            xcm,ycm = getCentreOfMass(r_in=r, m_in=self._sa.mff)
                            rcm = [-xcm, y-ycm, -x-xcm, -ycm, (D-x)-xcm, -ycm]
                            
                            # Initial velocities            
                            v = [0.0] * 6
                    
                            # Check that initial conditions are valid
                            initErrors += sim.checkConfiguration(r_in=r, v_in=v, TXE_in=0.0)
                            #sim = SimulateTrajectory(sa=self._sa, r_in=r, v_in=v, TXE=0.0)
                            #initErrors = sim.getExceptionCount()
                    
                            if (np.sum(Ec0) + self._Ekin0) > self._sa.Q:
                                initErrors += 1
                            
                            # Store initial conditions if they are good
                            if initErrors > 0:
                                badOnes += 1    
                            rs[s] = r
                            vs[s] = v
                            
                            if plotInitialConfigs:
                                Ds[s] = D
                                x_plot[s] = x
                                y_plot[s] = y
                                Ecs[s] = np.sum(Ec0)
                            
                            s+=1
                            
                            if s%5000 == 0 and s > 0 and verbose:
                                print(str(s)+" of "+str(self._sims)+" initial "
                                      "conditions generated.")
                        # end of x loop
                    # end of y loop
                # end of D loop
            elif self._mode == "uncertainty":
                thisError = 0
                thatError = 0
                someError = 0
                anyError = 0
                mu_p = [0.0,0.0,0.0]
                sigma_p = [[self._sigma_px, 0.0,            0.0],
                           [0.0,            self._sigma_py, 0.0],
                           [0.0,            0.0,            self._sigma_py]]
                while s < self._sims:
                    initErrors = 0
                    
                    #px, py_0, pz_0 = np.random.multivariate_normal(mu_p, sigma_p)
                    px = np.random.normal(0.0, self._sigma_px)
                    py_0 = np.random.normal(0.0, self._sigma_py)
                    pz_0 = np.random.normal(0.0, self._sigma_py)
                    ydir = np.sign(py_0)
                    ############
                    ################                      #####
                    py = np.sqrt(py_0**2 + pz_0**2)
                    #py = py_0
                    ######## #       ####
                    ########################      ######### 
                    #################
                    Ekin_tp = 0.5 / self._sa.mff[0] * (px**2 + py**2)
                    
                    
                    vtpx = px/self._sa.mff[0]
                    vtpy = py/self._sa.mff[0]
                    
                    # Randomize kinetic energy of fission fragments
                    Eff = np.random.normal(5.0, 1.0)
                    
                    _setDminDmax(self, Ekin0_in=(Eff+Ekin_tp))
                    D = np.random.random() * (self._Dmax - self._Dmin) + self._Dmin
                    A = ((self._sa.Q-self._minTol-Ekin_tp-Eff)/1.43996518-self._sa.Z[1]*self._sa.Z[2]/D)/self._sa.Z[0]
                    polyn = [A,(self._sa.Z[2]-self._sa.Z[1]-A*D),D*self._sa.Z[1]]
                    sols = np.roots(polyn)
                    
                    # Calculate xmin and xmax
                    xmin = max(sols[1],self._sa.ab[0]+self._sa.ab[2]+self._minTol)
                    xmax = min(sols[0],D-(self._sa.ab[0]+self._sa.ab[4]+self._minTol))
                    
                    if np.iscomplex(xmin):
                        thisError += 1
                        initErrors += 1
                    if np.iscomplex(xmax):
                        initErrors += 1
                        thisError += 1
                    # Randomize x
                    x = np.random.random() * (xmax - xmin) + xmin
                    y = 0
                    
                    r = [0,y,-x,0,D-x,0]
                    Ec0 = self._sa.cint.coulombEnergies(Z_in=self._sa.Z,r_in=r,fissionType_in=self._sa.fissionType)
                    
                    xcm,ycm = getCentreOfMass(r_in=r, m_in=self._sa.mff)
                    rcm = [-xcm, y-ycm, -x-xcm, -ycm, (D-x)-xcm, -ycm]
                    
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
                        thatError += 1
                        initErrors += 1
                        #print(Ekin_tp)
                        #raise ValueError('Complex root: '+str(sols)+' (Eav='+str(Eav-Ekin_tp-Eff)+')')
                    
                    # Verify that total lin. and ang. mom. is zero
                    ptotx = px + phfx + plfx
                    ptoty = py + phfy + plfy
                    #angmom = rcm[0]*py-rcm[1]*px + rcm[2]*phfy-rcm[3]*phfx + rcm[4]*plfy-rcm[5]*phfy
                    angmom = -y*px - x*phfy + (D-x)*plfy
                    if not np.allclose(ptotx,0.0):
                        someError += 1
                        initErrors += 1
                        #raise ValueError(str(i)+"Linear mom. not conserved: ptotx = "+str(ptotx)+"\tp: ["+str(px)+","+str(phfx)+","+str(plfx)+"]")
                    if not np.allclose(ptoty,0.0):
                        someError += 1
                        initErrors += 1
                        #raise ValueError(str(i)+"Linear mom. not conserved: ptoty = "+str(ptoty)+"\tp: ["+str(py)+","+str(phfy)+","+str(plfy)+"]")
                    if not np.allclose(angmom,0.0):
                        someError += 1
                        initErrors += 1
                        #raise ValueError(str(i)+"Angular mom. not conserved: angmom = "+str(angmom)+"\tp: ["+str(-y*px)+","+str(-x*phfy)+","+str((D-x)*plfy)+"]")
                    
                    # Calculate speeds of fission fragments
                    vhfx = phfx / self._sa.mff[1]
                    vhfy = phfy / self._sa.mff[1]
                    vlfx = plfx / self._sa.mff[2]
                    vlfy = plfy / self._sa.mff[2]
                    
                    # Initial velocities            
                    v = [vtpx,vtpy,vhfx,vhfy,vlfx,vlfy]
                    
                    if not np.allclose(0.0, (px + phfx + plfx)):
                        print px
                        print phfx
                        print plfx
                        raise Exception("Non-zero momenta.")
                   
                    # Check that initial conditions are valid
                    initErrors += sim.checkConfiguration(r_in=r, v_in=v, TXE_in=0.0)
                    
                    if (np.sum(Ec0) + Eff + Ekin_tp) > self._sa.Q:
                        anyError += 1
                        initErrors += 1
                    
                    # Store initial conditions if they are good
                    if initErrors > 0:
                        badOnes += 1
                    else:
                        rs[s] = r
                        vs[s] = v
                        
                        if plotInitialConfigs:
                            Ds[s] = D
                            x_plot[s] = x
                            y_plot[s] = y
                            Ecs[s] = np.sum(Ec0)
                        
                        s+=1
                        
                        if s%5000 == 0 and s > 0 and verbose:
                            print(str(s)+" of "+str(self._sims)+" initial "
                                  "conditions generated.")
                            #h = hpy()
                            #print h.heap()
                # end of for loop
            elif self._mode == "triad":
                y_linspace = np.linspace(0.0, self._yMax, int(self._sims/3))
                Ds = [self._Dmin, self._D_tpl_contact, self._D_tph_contact]
                xs = [(self._xsaddle*self._Dmin),
                      (Ds[1]-self._dcontact-self._minTol),
                      (self._xcontact+self._minTol)]
                s = 0
                print Ds
                print xs
                for i in range(0,int(self._sims/3)):
                    for j in range(0,3):
                        D = Ds[j]
                        x = xs[j]
                        
                        y = 0
                        
                        r = [0,y,-x,0,D-x,0]
                        v = [0] * 6
                            
                        rs[s] = r
                        vs[s] = v
                            
                        s += 1
                        if s%5000 == 0 and s > 0 and verbose:
                            print(str(s)+" of "+str(self._sims)+" initial "
                              "conditions generated.")
                    #end of inner loop
                #end of for loop 
            
            print("-----------------------------------------------------------")
            print(thisError)
            print(thatError)
            print(someError)
            print(anyError)
            print(str(badOnes)+" out of "+str(self._sims)+" simulations failed.")
            
            # Save initial configurations
            if self._saveConfigs and (self._oldConfigs == None or not os.path.isfile(self._oldConfigs)):
                
                # Create results folder if it doesn't exist
                if not os.path.exists(filePath):
                    os.makedirs(filePath)
                
                # Store static variables in a shelved file format
                s = shelve.open(filePath + "initialConfigs.sb", protocol=pickle.HIGHEST_PROTOCOL)
                try:
                    s["0"] = { 'r0': rs, 'v0': vs, 'sims': self._sims}
                finally:
                    s.close()
                """
                f_data = file(filePath + "initialConfigs.bin", 'wb')
                np.save(f_data,np.array([rs, vs, TXE, self._sims]))
                f_data.close()
                """
                
                if verbose:
                    print("Saved initial configs to: "+ filePath + "initialConfigs.sb")
        else:
            sv = shelve.open(self._oldConfigs, protocol=pickle.HIGHEST_PROTOCOL)
            for row in sv:
                rs = sv[row]['r0']
                vs = sv[row]['v0']
                self._sims = sv[row]['sims']
                try:
                    TXE = sv[row]['TXE']
                except KeyError:
                    TXE = np.zeros(self._sims)
            """
            with open(self._oldConfigs,"rb") as f_data:
                npdata = np.load(f_data)
                rs = npdata[0]
                vs = npdata[1]
                TXE = npdata[2]
                self._sims = npdata[3]
            """
            Ecs = [0]*self._sims
            for ri in range(0,self._sims):
                Ecs[ri] = np.sum(self._sa.cint.coulombEnergies(Z_in=self._sa.Z,r_in=rs[ri],fissionType_in=self._sa.fissionType))
            if verbose:
                print("Loaded "+str(self._sims)+" intial configurations "
                      "from: "+self._oldConfigs)        
        if plotInitialConfigs:
            fig = plt.figure(1)
            ax = fig.add_subplot(111)
            ny, binsy, patches = ax.hist(Ecs, bins=100)
            bincentersy = 0.5*(binsy[1:]+binsy[:-1])
            l = ax.plot(bincentersy, ny, 'r--', linewidth=4,label='Ec0')
            ax.set_title('Initial Coulomb Energy, Q = %1.1f' % (self._sa.Q))
            ax.set_xlabel('Ec0 [MeV]')
            ax.set_ylabel('Counts')
            #ax.set_xlim([0,14.001])
            
            fig = plt.figure(2)
            ax = fig.add_subplot(111)
            ny, binsy, patches = ax.hist(Ds, bins=100)
            bincentersy = 0.5*(binsy[1:]+binsy[:-1])
            l = ax.plot(bincentersy, ny, 'r--', linewidth=4,label='Ec0')
            ax.set_title("D")
            ax.set_xlabel('D [fm]')
            ax.set_ylabel('Counts')
            
            H, xedges, yedges = np.histogram2d(x_plot,y_plot,bins=200)
            # H needs to be rotated and flipped
            H = np.rot90(H)
            H = np.flipud(H)
            Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
            fig = plt.figure(3)
            ax = fig.add_subplot(111)
            plt.pcolormesh(xedges,yedges,Hmasked)
            plt.title("xy")
            plt.xlabel("x")
            plt.ylabel("y")
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Counts')
            
            plt.show()
        
        return rs, vs, TXE, self._sims


        
    def oldShiz(self):
        minTol = 0.0001
        
        dcount = 0
        
        simulationNumber = 0
        totSims = 0
        
        if self._config == 'max':
            ekins = np.linspace(0,self._Ekin0,self._sims)
            D = []
            for i in range(0,self._sims):
                Eav = self._sa.Q - ekins[i] - minTol
                A = ((Eav)/1.43996518 - self._sa.Z[0]*self._sa.Z[1]/xcontact)/self._sa.Z[2]
                D_tph_contact = np.float(nsolve(Dsym**2*A - \
                                                Dsym*(A*xcontact + self._sa.Z[0] + 
                                                      self._sa.Z[1]) + \
                                                self._sa.Z[1]*xcontact, Dsym, 30.0))
                """Dmin = np.float(nsolve(Dsym - (self._sa.Z[0]*self._sa.Z[1]/xsaddle + \
                                               self._sa.Z[0]*self._sa.Z[2]/dsaddle + \
                                               self._sa.Z[1]*self._sa.Z[2] \
                                              )*1.43996518/(Eav), Dsym, 18.0))
                D.append(Dmin)
                xs.append([xsaddle*Dmin])"""
                D.append(D_tph_contact)
            xs = [[xcontact+minTol]]*self._sims
            ys = [0]*self._sims
            print D[0]
            print D[-1]
            ekins2 = np.linspace(0,192,self._sims)
            import matplotlib.pyplot as plt
            #plt.plot(ekins2, ekins2/(self._sa.mff[0]/self._sa.mff[1] + 1.0))
            #plt.plot(ekins2, ekins2/(self._sa.mff[1]/self._sa.mff[0] + 1.0))
            plt.plot(ekins,D)
            plt.show()
        elif self._config == 'triad':
            D = [Dmin, D_tpl_contact, D_tph_contact]
            xs = [[xsaddle*Dmin], [D[1]-dcontact-minTol], [xcontact+minTol]]
            ys = [0]*3
            print D
        else:
            D = np.linspace(Dmin, Dmax, self._sims)
        
            xs = [0]*self._sims
            ys = [0]*self._sims
            
            
            xs[0] = [xsaddle*D[0]]
            
            for i in range(1,len(D)):
                A = ((self._sa.Q-minTol)/1.43996518-self._sa.Z[1]*self._sa.Z[2]/D[i])/self._sa.Z[0]
                p = [A,(self._sa.Z[2]-self._sa.Z[1]-A*D[i]),D[i]*self._sa.Z[1]]
                sols = np.roots(p)
                #print(str(D[i])+'\t'),
                #print(sols),
                
                # Check 2 sols
                if len(sols) != 2:
                    raise Exception('Wrong amount of solutions: '+str(len(sols)))
                # Check reals
                #if np.iscomplex(sols[0]):
                #    raise ValueError('Complex root: '+str(sols[0]))
                #if np.iscomplex(sols[1]):
                #    raise ValueError('Complex root: '+str(sols[1]))
                
                xmin = max(sols[1],self._sa.ab[0]+self._sa.ab[2]+minTol)
                xmax = min(sols[0],D[i]-(self._sa.ab[0]+self._sa.ab[4]+minTol))
                
                """
                s1 = '_'
                s2 = '_'
                if xmin == self._sa.ab[0]+self._sa.ab[2]+minTol:
                    s1 = 'o'
                if xmax == D[i]-(self._sa.ab[0]+self._sa.ab[4]+minTol):
                    s2 = 'o'
                print(s1+s2)
                """
                
                # Check inside neck
                if xmin < (self._sa.ab[0]+self._sa.ab[2]):
                    raise ValueError('Solution overlapping with HF:')
                if xmin > D[i]-(self._sa.ab[0]+self._sa.ab[4]):
                    raise ValueError('Solution overlapping with LF:')
                if xmax < (self._sa.ab[0]+self._sa.ab[2]):
                    raise ValueError('Solution overlapping with HF:')
                if xmax > D[i]-(self._sa.ab[0]+self._sa.ab[4]):
                    raise ValueError('Solution overlapping with LF:')
                
                xs[i] = np.arange(xmin,xmax,self._dx)
        
        # Assign y-values
        for i in range(0,len(D)):
            ys[i] = [0]*len(xs[i])
            for j in range(0,len(xs[i])):
                if self._yMax == 0 or self._dy == 0:
                    ys[i][j] = [0]
                else:
                    ys[i][j] = np.arange(0,self._yMax,self._dy)
        
        # Count total simulations
        for i in range(0,len(D)):
            for j in range(0,len(xs[i])):
                    totSims += len(ys[i][j])
        
        ENi_max = 0
        for i in range(0,len(D)):
            for j in range(0,len(xs[i])):
                for k in range(0,len(ys[i][j])):
                    simulationNumber += 1
                    r = [0.0,ys[i][j][k],-xs[i][j],0.0,D[i]-xs[i][j],0.0]
                    
                    v = [0.0]*6
                    
                    if self._config == 'max':
                        v[0] = np.sqrt(2*ekins[i]/(self._sa.mff[0]**2/self._sa.mff[1] + self._sa.mff[0]))
                        v[2] = -np.sqrt(2*ekins[i]/(self._sa.mff[1]**2/self._sa.mff[0] + self._sa.mff[1]))
                    
                    sim = SimulateTrajectory(sa=self._sa, r_in=r, v_in=v)
                    e, outString, ENi = sim.run(simulationNumber=simulationNumber, timeStamp=timeStamp)
                    if ENi > ENi_max:
                        ENi_max = ENi
                    #sim.plotTrajectories()
                    if e == 0:
                        print("S: "+str(simulationNumber)+"/~"+str(totSims)+"\t"+str(r)+"\t"+outString)
        print("Total simulation time: "+str(time()-simTime)+"sec")
        print("Maximum Nickel Energy: "+str(ENi_max)+" MeV")

def _setDminDmax(self, Ekin0_in):
    # Calculate limits
    Eav = self._sa.Q - np.sum(Ekin0_in) - self._minTol
    
    # Solve Dmin
    Dsym = Symbol('Dsym')
    self._Dmin = np.float(nsolve(Dsym - (self._sa.Z[0]*self._sa.Z[1]/self._xsaddle + \
                                         self._sa.Z[0]*self._sa.Z[2]/self._dsaddle + \
                                         self._sa.Z[1]*self._sa.Z[2] \
                                        )*1.43996518/(Eav), Dsym, 18.0)) + \
                 self._deltaDmin + self._minTol
    
    self._Dmax = self._Dmin + self._deltaDmax
    
    # What is minimum D when TP and LF are touching
    A = ((Eav)/1.43996518 - self._sa.Z[0]*self._sa.Z[2]/self._dcontact)/self._sa.Z[1]
    self._D_tpl_contact = np.float(nsolve(Dsym**2*A - \
                                          Dsym*(A*self._dcontact + self._sa.Z[0] + 
                                                self._sa.Z[2]) + \
                                          self._sa.Z[2]*self._dcontact, Dsym, 26.0))
    # What is minimum D when TP and HF are touching
    A = ((Eav)/1.43996518 - self._sa.Z[0]*self._sa.Z[1]/self._xcontact)/self._sa.Z[2]
    self._D_tph_contact = np.float(nsolve(Dsym**2*A - \
                                          Dsym*(A*self._xcontact + self._sa.Z[0] + 
                                                self._sa.Z[1]) + \
                                          self._sa.Z[1]*self._xcontact, Dsym, 30.0))
    del A
    del Dsym
    del Eav


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
from time import time
from datetime import datetime
import matplotlib.pyplot as plt
        
class RegionGenerator:
    """
    """
    
    def __init__(self, sa, sims, mu_D, mode, lineSamples = 500, saveConfigs = False, oldConfigs = None):
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
        self._saveConfigs = saveConfigs
        self._oldConfigs = oldConfigs
        self._mu_D = mu_D
        self._mode = mode
        self._lineSamples = lineSamples
        
        self._minTol = 0.0001
            
        if self._mode == "randD":
            self._dr = 1/(np.sqrt(self._sa.Z[1]/self._sa.Z[2]) + 1.0)
            self._xr = 1.0 - self._dr

            self._mu_D = self._sa.cint.solveDwhenTPonAxis(xr_in=self._xr,
                                                          E_in=E_solve,
                                                          Z_in=self._sa.Z,
                                                          sol_guess=21.0)
        print("Mu_D = "+str(self._mu_D))
        
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
                vx_plot = [0]*self._sims
                vy_plot = [0]*self._sims
                vy2_plot = [0]*self._sims
                
                ekin_plot = [0]*self._sims
                ekinh_plot = [0]*self._sims
                ekinl_plot = [0]*self._sims
                etot_plot = [0]*self._sims
                Ds = [0]*self._sims
                costhetas = [0]*self._sims
                sinthetas = [0]*self._sims
                Erot = [0]*self._sims
            
            
            rs = np.zeros([self._sims,6])
            vs = np.zeros([self._sims,6])
            TXE = np.zeros(self._sims)
            
            i = 0
            tries = 0
            thisError = 0
            
            prevD = None
            
            while i < self._sims:
                tries += 1
                initErrors = 0
                
                # Get D
                if self._mode == "randD" and tries > 1:
                    print('Not yet done')
                else:
                    D = self._mu_D
                
                if D != prevD:
                    xl,ylQ,ylQf = getClosestConfigurationLine(D_in = D,
                                                              Dsize_in = self._lineSamples,
                                                              E_in = (self._sa.Q),
                                                              Z_in = self._sa.Z,
                                                              pint_in = self._sa.cint,
                                                              ab_in = self._sa.ab)
                    prevD = D
                    
                # Randomize x
                x = np.random.random() * D
                
                xClosest, xClosestI = None, None
                
                for lower,upper in zip(xl[:-1],xl[1:]):
                    if lower <= x <= upper:
                        xClosest = lower
                        xClosestI = np.where(xl==lower)[0][0]
                
                # Randomize y
                ymin = ylQf[xClosestI]
                ymax = max(ylQf) + 2.0
                
                y = np.random.random() * (ymax - ymin) + ymin 
                
                # Start positions
                r = [0,y,-x,0,D-x,0]
                
                Ec0 = self._sa.cint.coulombEnergies(Z_in=self._sa.Z,r_in=r,fissionType_in=self._sa.fissionType)
                Eav = self._sa.Q - np.sum(Ec0)
                Eff = 0
                Ekin_tp = 0
                # Get Center of Mass coordinates
                xcm,ycm = getCentreOfMass(r_in=r, m_in=self._sa.mff)
                rcm = [-xcm, y-ycm, -x-xcm, -ycm, (D-x)-xcm, -ycm]
                
                """
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
                    initErrors += 1
                    #print(Ekin_tp)
                    #raise ValueError('Complex root: '+str(sols)+' (Eav='+str(Eav-Ekin_tp-Eff)+')')
                
                # Verify that total lin. and ang. mom. is zero
                ptotx = px + phfx + plfx
                ptoty = py + phfy + plfy
                #angmom = rcm[0]*py-rcm[1]*px + rcm[2]*phfy-rcm[3]*phfx + rcm[4]*plfy-rcm[5]*phfy
                angmom = -y*px - x*phfy + (D-x)*plfy
                if not np.allclose(ptotx,0.0):
                    initErrors += 1
                    #raise ValueError(str(i)+"Linear mom. not conserved: ptotx = "+str(ptotx)+"\tp: ["+str(px)+","+str(phfx)+","+str(plfx)+"]")
                if not np.allclose(ptoty,0.0):
                    initErrors += 1
                    #raise ValueError(str(i)+"Linear mom. not conserved: ptoty = "+str(ptoty)+"\tp: ["+str(py)+","+str(phfy)+","+str(plfy)+"]")
                if not np.allclose(angmom,0.0):
                    initErrors += 1
                    #raise ValueError(str(i)+"Angular mom. not conserved: angmom = "+str(angmom)+"\tp: ["+str(-y*px)+","+str(-x*phfy)+","+str((D-x)*plfy)+"]")
                
                # Calculate speeds of fission fragments
                vhfx = phfx / self._sa.mff[1]
                vhfy = phfy / self._sa.mff[1]
                vlfx = plfx / self._sa.mff[2]
                vlfy = plfy / self._sa.mff[2]
                """
                vtpx = 0
                vtpy = 0
                vhfx = 0
                vhfy = 0
                vlfx = 0
                vlfy = 0
                
                # Initial velocities            
                v = [vtpx,vtpy,vhfx,vhfy,vlfx,vlfy]
                                
                # Check that initial conditions are valid
                initErrors += sim.checkConfiguration(r_in=r, v_in=v, TXE_in=0.0)
                #sim = SimulateTrajectory(sa=self._sa, r_in=r, v_in=v, TXE=0.0)
                #initErrors = sim.getExceptionCount()
                
                if (Eff + Ekin_tp + np.sum(Ec0)) > self._sa.Q:
                    initErrors += 1
                
                # Store initial conditions if they are good
                if initErrors == 0:

                    rs[i] = r
                    vs[i] = v
                    
                    if plotInitialConfigs:
                        Ds[i] = D
                        costhetas[i] = costheta
                        sinthetas[i] = sintheta
                        x_plot[i] = x
                        y_plot[i] = y
                        vx_plot[i] = vtpx/0.011
                        vy_plot[i] = vtpy/0.011
                        vy2_plot[i] = py2/self._sa.mff[0]/0.011
                        ekin_plot[i] = Ekin_tp
                        ekinh_plot[i] = 0.5*(phfx**2 + phfy**2)/self._sa.mff[1]
                        ekinl_plot[i] = 0.5*(plfx**2 + plfy**2)/self._sa.mff[2]
                        etot_plot[i] = Ekin_tp + Eff + np.sum(Ec0)

                    i += 1
                    if i%5000 == 0 and i > 0 and verbose:
                        print(str(i)+" of "+str(self._sims)+" initial "
                              "conditions generated.")
            
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
            if verbose:
                print("Loaded "+str(self._sims)+" intial configurations "
                      "from: "+self._oldConfigs)        
        if plotInitialConfigs:
            print('Plot initial configs functionality not done yet.')
            
        return rs, vs, TXE, self._sims
    
    def oldShiz(self):
        xl,ylQ,ylQf = getClosestConfigurationLine(self._D,500,(self._sa.Q+self._dE),self._sa.Z,self._sa.cint,self._sa.ab)
        """
        for i in range(0,len(ylQ)):
            ylQ[i] = self._sa.cint.solvey(D_in=self._D, x_in=xl[i], E_in=(self._Q+self._dE), Z_in=self._sa.Z, sol_guess=10.0)
            
            if (self._D-(self._sa.ab[0]+self._sa.ab[4])) < xl[i] < (self._sa.ab[0]+self._sa.ab[2]):
                ylQf[i] = np.max([(self._sa.ab[3]+self._sa.ab[1])*np.sqrt(1.0-(xl[i]/(self._sa.ab[2]+self._sa.ab[0]))**2),
                                  (self._sa.ab[5]+self._sa.ab[1])*np.sqrt(1.0-((self._D-xl[i])/(self._sa.ab[4]+self._sa.ab[0]))**2),
                                  ylQ[i]])
            elif xl[i] < (self._sa.ab[0]+self._sa.ab[2]) and xl[i] < (self._D-(self._sa.ab[0]+self._sa.ab[4])):
                ylQf[i] = np.max([(self._sa.ab[3]+self._sa.ab[1])*np.sqrt(1.0-(xl[i]/(self._sa.ab[2]+self._sa.ab[0]))**2),ylQ[i]])
            elif xl[i] > (self._D-(self._sa.ab[0]+self._sa.ab[4])) and xl[i] > (self._sa.ab[0]+self._sa.ab[2]):
                ylQf[i] = np.max([(self._sa.ab[5]+self._sa.ab[1])*np.sqrt(1.0-((self._D-xl[i])/(self._sa.ab[4]+self._sa.ab[0]))**2),ylQ[i]])
            else:
                ylQf[i] = ylQ[i]
        """
        """
            ylQ[i] = self._sa.cint.solvey(D_in=self._D, x_in=xl[i], E_in=(self._Q+self._dE), Z_in=self._sa.Z, sol_guess=10.0)
            
            if xl[i]<self._sa.rad[0]+self._sa.rad[1]:
                ylQf[i] = max(np.sqrt((self._sa.rad[0]+self._sa.rad[1])**2-xl[i]**2),ylQ[i])
            elif xl[i]>(self._D-(self._sa.rad[0]+self._sa.rad[2])):
                ylQf[i] = max(np.sqrt((self._sa.rad[0]+self._sa.rad[2])**2-(self._D-xl[i])**2),ylQ[i])
            else:
                ylQf[i] = ylQ[i]
            #print('('+str(xl[i])+','+str(ylQf[i])+')')
        """
        
        xStart = self._sa.ab[2]*1.0
        xStop = self._D-self._sa.ab[4]*1.0
        xLow, xHigh, xLowI, xHighI = None, None, None, None
        
        for lower,upper in zip(xl[:-1],xl[1:]):
            if lower <= xStart <= upper:
                xLow = lower
                xLowI = np.where(xl==lower)[0][0]
            if lower <= xStop <= upper:
                xHigh = upper
                xHighI = np.where(xl==upper)[0][0]
        
        if xLow == None:
            raise Exception('Couldnt find xLow')
        if xHigh == None:
            raise Exception('Couldnt find xHigh')
        if xLowI == None:
            raise Exception('Couldnt find xLowI')
        if xHighI == None:
            raise Exception('Couldnt find xHighI')
        
        randx = xl[xLowI:xHighI]
        ys = ylQf[xLowI:xHighI] # These will be our yLow:s
        simulationNumber = 0
        
        #yHigh = self._sa.ab[3]-self._sa.ab[1]#np.max(ylQf)
        yHigh = max(ylQf[xLowI],ylQf[xHighI])
        
        totSims = 0
        
        for yc in ys:
            totSims += len(np.arange(yc,yHigh,0.05))
        self._sims = totSims
        
        for i in range(0,len(randx)):
            randy = np.arange(ys[i],yHigh,0.05)
            for j in range(0,len(randy)):
                simulationNumber += 1
                x = randx[i]
                y = randy[j]
                r = [0,y,-x,0,self._D-x,0]
                
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


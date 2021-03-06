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
        
class GeneratorThree:
    """
    Generate uniform initial configurations in the neck region, based on the
    closest possible distance.
    """
    
    def __init__(self, sa, sims, D, dx, dy, dE):
        """
        Initialize and pre-process the simulation data.

        :type sa: :class:`tftsim_args.TFTSimArgs` class
        :param sa: An instance of TFTSimArgs describing what kind of system to
                   simulate.
        
        :type sims: int
        :param sims: Number of simulations to run.
        
        :type DMu: float
        :param DMu: Center of the gaussian probability for D, measured from
                    Dmin in fm.
        
        :type DSigma: float
        :param DSigma: Spread of the gaussian probability for D in fm,
                       corresponding to the FWHM.
        
        :type yMu: float
        :param yMu: Center of the gaussian probability for y. Measured from
                    ymin in fm.
        
        :type ySigma: float
        :param ySigma: Spread of the gaussian probability for y in fm,
                       corresponding to the FWHM.
        
        :type ymin: float
        :param ymin: Minimum displacement of ternary particle from fission axis
                     in fm. A value of zero is generally not considered.
        """
        
        # Check for exceptions in input parameters

        self._sa = sa
        self._sims = sims
        self._D = D
        self._dx = dx
        self._dy = dy
        self._dE = dE
        
        self._minTol = 0.1
        
        # Failsafes that makes returns an exception if this procedure is incompatible with the fissioning system 
        #if derp:
        #   raise Exception('herp')
        """
        xl = np.linspace(0.0,self._D,100)
        ylQ = np.zeros_like(xl)
        ylQf = np.zeros_like(xl)
        for i in range(0,len(ylQ)):
            ylQ[i] = self._sa.cint.solvey(self._D, xl[i], self._sa.Q+self._dE, self._sa.Z, 10.0)
            
            if xl[i]<self._sa.rad[0]+self._sa.rad[1]:
                ylQf[i] = max(np.sqrt((self._sa.rad[0]+self._sa.rad[1])**2-xl[i]**2),ylQ[i])
            elif xl[i]>(self._D-(self._sa.rad[0]+self._sa.rad[2])):
                ylQf[i] = max(np.sqrt((self._sa.rad[0]+self._sa.rad[2])**2-(self._D-xl[i])**2),ylQ[i])
            else:
                ylQf[i] = ylQ[i]
            #print('('+str(xl[i])+','+str(ylQf[i])+')')
        """
        ########################################################################
        """    
        fig = plt.figure(1)
        xs = np.array([0,self._D])
        ys = np.array([0,0])
        rs = np.array([self._sa.rad[1],self._sa.rad[2]])

        phi = np.linspace(0.0,2*np.pi,100)

        na=np.newaxis

        # the first axis of these arrays varies the angle, 
        # the second varies the circles
        x_line = xs[na,:]+rs[na,:]*np.sin(phi[:,na])
        y_line = ys[na,:]+rs[na,:]*np.cos(phi[:,na])

        plt.plot(x_line,y_line,'-', linewidth=3.0)
        
        plt.plot(xl, ylQ, 'r--', linewidth=3.0, label='E = Q')
        plt.plot(xl, ylQf, 'b-', linewidth=3.0, label='E = Q, non-overlapping radii')
        plt.text(0,0, str('HF'),fontsize=20)
        plt.text(self._D,0, str('LF'),fontsize=20)
        plt.show()
        """
        
    def generate(self):
        """
        Generate initial configurations.
        """  
        
        dcount = 0
        simTime = time()
        timeStamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
        
        
        #xl = np.linspace(0.0,self._D,500)
        #ylQ = np.zeros_like(xl)
        #ylQf = np.zeros_like(xl)
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


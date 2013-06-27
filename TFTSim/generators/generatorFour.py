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
        
class GeneratorFour:
    """

    """
    
    def __init__(self, sa, sims, D, dx, dy, dE, angles, radii):
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

        :type angles: int
        :param angles: Number of angles to use in each starting spot.
        
        :type radii: int
        :param radii: Number of radii to use in each starting spot.
        
        """
        
        # Check for exceptions in input parameters

        self._sa = sa
        self._sims = sims
        self._D = D
        self._dx = dx
        self._dy = dy
        self._dE = dE
        self._Z = [self._sa.tp.Z, self._sa.hf.Z, self._sa.lf.Z]
        self._rad = [crudeNuclearRadius(self._sa.tp.A),
                     crudeNuclearRadius(self._sa.hf.A),
                     crudeNuclearRadius(self._sa.lf.A)]
        self._mff = [u2m(self._sa.tp.A), u2m(self._sa.hf.A), u2m(self._sa.lf.A)]
        self._minTol = 0.1
        self._angles = angles
        self._radii = radii

        self._Q = getQValue(self._sa.fp.mEx,self._sa.pp.mEx,self._sa.tp.mEx,self._sa.hf.mEx,self._sa.lf.mEx,self._sa.lostNeutrons)
        
        if not self._angles > 0 or not isinstance(self._angles, int):
            raise ValueError('angles must be an integer greater than zero.')
        if not self._radii > 0 or not isinstance(self._radii, int):
            raise ValueError('radii must be an integer greater than zero.')
        
        # Failsafes that makes returns an exception if this procedure is incompatible with the fissioning system 
        #if derp:
        #   raise Exception('herp')
        """
        xl = np.linspace(0.0,self._D,100)
        ylQ = np.zeros_like(xl)
        ylQf = np.zeros_like(xl)
        for i in range(0,len(ylQ)):
            ylQ[i] = self._sa.pint.solvey(self._D, xl[i], self._Q+self._dE, self._Z, 10.0)
            
            if xl[i]<self._rad[0]+self._rad[1]:
                ylQf[i] = max(np.sqrt((self._rad[0]+self._rad[1])**2-xl[i]**2),ylQ[i])
            elif xl[i]>(self._D-(self._rad[0]+self._rad[2])):
                ylQf[i] = max(np.sqrt((self._rad[0]+self._rad[2])**2-(self._D-xl[i])**2),ylQ[i])
            else:
                ylQf[i] = ylQ[i]
            #print('('+str(xl[i])+','+str(ylQf[i])+')')
        """
        ########################################################################
        """    
        fig = plt.figure(1)
        xs = np.array([0,self._D])
        ys = np.array([0,0])
        rs = np.array([self._rad[1],self._rad[2]])

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
        
        
        xl = np.linspace(0.0,self._D,100)
        ylQ = np.zeros_like(xl)
        ylQf = np.zeros_like(xl)
        for i in range(0,len(ylQ)):
            ylQ[i] = self._sa.pint.solvey(D_in=self._D, x_in=xl[i], E_in=(self._Q+self._dE), Z_in=self._Z, sol_guess=10.0)
            
            if xl[i]<self._rad[0]+self._rad[1]:
                ylQf[i] = max(np.sqrt((self._rad[0]+self._rad[1])**2-xl[i]**2),ylQ[i])
            elif xl[i]>(self._D-(self._rad[0]+self._rad[2])):
                ylQf[i] = max(np.sqrt((self._rad[0]+self._rad[2])**2-(self._D-xl[i])**2),ylQ[i])
            else:
                ylQf[i] = ylQ[i]
            #print('('+str(xl[i])+','+str(ylQf[i])+')')
        
        xStart = self._rad[1]*0.0
        xStop = self._D-self._rad[2]*0.0
        xLow, xHigh, xLowI, xHighI = None, None, None, None
        yHigh = np.max(ylQf)+0.5
        
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
        
        totSims = 0
        
        for yc in ys:
            totSims += len(np.arange(yc,yHigh,0.5))
        self._sims = totSims*self._radii*self._angles
        
        angles_use = np.linspace(0,2.0*np.pi*float(self._angles)/float(self._angles+1),self._angles)
        for i in range(0,len(randx)):
            randy = np.arange(ys[i],yHigh,0.5)
            for j in range(0,len(randy)):
                x = randx[i]
                y = randy[j]
                r = [0,y,-x,0,self._D-x,0]
                
                Ec0 = self._sa.pint.coulombEnergies(self._Z, r)
                Eav = self._Q - np.sum(Ec0)
                vtot = np.linspace(0,np.sqrt(2*Eav/self._mff[0]),self._radii)
                etot = np.linspace(0,Eav,self._radii)
                for k in range(0,self._angles):
                    for l in range(0,self._radii):
                        
                        simulationNumber += 1
                        
                        
                        v1 = vtot[l]*np.cos(angles_use[k])
                        v2 = vtot[l]*np.sin(angles_use[k])
                        
                        # Randomize a kinetic energy for the ternary particle
                        #Ekin_tot = np.random.uniform(0.0,Eav*0.9)
                        
                        # Randomize initial direction for the initial momentum of tp
                        """
                        p2 = Ekin_tot * 2 * self._mff[0]
                        px2 = np.random.uniform(0.0,p2)
                        py2 = p2 - px2
                        xdir = np.random.randint(2)*2 - 1
                        ydir = np.random.randint(2)*2 - 1
                        """
                        
                        #v = [xdir*np.sqrt(px2)/self._mff[0],np.sqrt(py2)/self._mff[0],0.0,0.0,0.0,0.0]
                        v = [v1,v2,0,0,0,0]
                        
                        sim = SimulateTrajectory(sa=self._sa, r=r, v=v)
                        e, outString = sim.run(simulationNumber=simulationNumber, timeStamp=timeStamp)
                        if e == 0:
                            print("S: "+str(simulationNumber)+"/~"+str(self._sims)+"\t"+str(r)+"\t"+outString)
        print("Total simulation time: "+str(time()-simTime)+"sec")


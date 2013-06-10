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
from TFTSim.tftsim_utils import *
import sys
import numpy as np
from scipy.integrate import odeint
from time import time
from time import sleep
from datetime import datetime
from collections import defaultdict
import os
import copy
import shelve
import pylab as pl
import matplotlib.pyplot as plt
import math

class SimulateTrajectory:
    """
    Initialization of simulates the trajectories for a given system. Preform
    the actual simulation with SimulateTrajectory.run().
    """

    def __init__(self, sa, r, v):
        """
        Pre-process and initialize the simulation data.

        :type sa: :class:`tftsim_args.TFTSimArgs` class
        :param sa: An instance of TFTSimArgs describing what kind of system to
                   simulate.
                   
        :type r: list of floats
        :param r: Coordinates of the fission fragments [xtp, ytp, xhf, yhf, xlf, ylf].
        
        :type r: list of floats
        :param r: Velocities of the fission fragments [vxtp, vytp, vxhf, vyhf, vxlf, vylf].
        """

        self._r = r
        self._v = v
        self._simulationName = sa.simulationName
        self._fissionType = sa.fissionType
        self._pint = copy.copy(sa.pint)
        self._fp = copy.copy(sa.fp)
        self._pp = copy.copy(sa.pp)
        self._tp = copy.copy(sa.tp)
        self._hf = copy.copy(sa.hf)
        self._lf = copy.copy(sa.lf)
        self._lostNeutrons = sa.lostNeutrons
        self._minEc = sa.minEc
        self._maxRunsODE = sa.maxRunsODE
        self._maxTimeODE = sa.maxTimeODE
        self._neutronEvaporation = sa.neutronEvaporation
        self._verbose = sa.verbose
        self._interruptOnException = sa.interruptOnException
        self._saveTrajectories = sa.saveTrajectories
        self._mff = [u2m(self._tp.A), u2m(self._hf.A), u2m(self._lf.A)]
        self._Z = [self._tp.Z, self._hf.Z, self._lf.Z]
        self._rtp = crudeNuclearRadius(self._tp.A)
        self._rhf = crudeNuclearRadius(self._hf.A)
        self._rlf = crudeNuclearRadius(self._lf.A)
        self._exceptionCount = 0
        self._exceptionMessage = None
        
        # Check that simulationName is a valid string
        if not isinstance(self._simulationName, basestring):
            _throwException(self,'TypeError', 'simulationName must be a string.')
        if self._simulationName == None or self._simulationName == '':
            _throwException(self,'ValueError', 'simulationName must be set to a non-empty string.')
            
        # Check that fissionType is a valid string
        if not isinstance(self._fissionType, basestring):
            _throwException(self,'TypeError', 'fissionType must be a string.')
        if self._fissionType == None or self._fissionType == '':
            _throwException(self,'ValueError', 'fissionType must be set to a non-empty string.')
        else:
            if self._fissionType not in ['LCP','CCT','BF']:
                _throwException(self,'ValueError','Invalid fissionType, must be one of LCP, CCT, BF.')
        
        # Check that lost neutron number is in proper format
        if not isinstance(self._lostNeutrons, int):
            _throwException(self,'TypeError', 'lostNeutrons must be an integer.')
        if self._lostNeutrons == None or self._lostNeutrons < 0:
            _throwException(self,'Exception','lostNeutrons must be set to a value >= 0.')

        # Check that particle number is not bogus
        if (self._lostNeutrons + self._tp.A + self._hf.A + self._lf.A) > (self._fp.A + self._pp.A):
            _throwException(self,'Exception',"Higher A coming out of fission than in! "+\
                            str(self._fp.A)+"+"+str(self._pp.A)+" < "+\
                            str(self._lostNeutrons)+"+"+str(self._tp.A)+"+"+\
                            str(self._hf.A)+"+"+str(self._lf.A))
        if (self._lostNeutrons + self._tp.A + self._hf.A + self._lf.A) < (self._fp.A + self._pp.A):
            _throwException(self,'Exception',"Higher A coming into fission than out! "+\
                            str(self._fp.A)+"+"+str(self._pp.A)+" > "+\
                            str(self._lostNeutrons)+"+"+str(self._tp.A)+"+"+\
                            str(self._hf.A)+"+"+str(self._lf.A))

        # Check that particles are correctly ordered after increasing size
        if self._tp.A > self._hf.A:
            _throwException(self,'Exception',"Ternary particle is heavier than the heavy fission"
                            " fragment!")
        if self._tp.A > self._lf.A:
            _throwException(self,'Exception',"Ternary particle is heavier than the light fission"
                            " fragment!")
        if self._lf.A > self._hf.A:
            _throwException(self,'Exception',"Light fission fragment is heavier than the heavy "
                            "fission fragment!")

        # Check that minEc is in proper format
        if not isinstance(self._minEc, float):
            _throwException(self,'TypeError','minEc needs to be a float.')
        if self._minEc == None or self._minEc <= 0 or self._minEc >= 1:
            _throwException(self,'Exception','minEc must be set to a value > 0 and < 1.')

        # Check that maxRunsODE is in proper format
        if not isinstance(self._maxRunsODE, int):
            _throwException(self,'TypeError','maxRunsODE needs to be an int.')
        if self._maxRunsODE == None or self._maxRunsODE < 0 or not np.isfinite(self._maxRunsODE):
            _throwException(self,'Exception',"maxRunsODE must be set to a finite value >= 0."
                                             "(0 means indefinite runs until convergence)")
        # Check that maxTimeODE is in proper format
        if not isinstance(self._maxTimeODE, int) and not isinstance(self._maxTimeODE, float):
            _throwException(self,'TypeError','maxRunsODE needs to be float or int.')
        if self._maxRunsODE == None or self._maxRunsODE < 0 or not np.isfinite(self._maxTimeODE):
            _throwException(self,'Exception',"maxRunsODE must be set to a finite value >= 0."
                                             "(0 means indefinite run time until until convergence)")

        self._Ec = self._pint.coulombEnergies(self._Z, self._r)

        self._Ekin = getKineticEnergies(self)

        # Check that Ec is a number

        # Check that minEc is not too high
        if self._minEc >= np.sum(self._Ec):
            _throwException(self,'Exception',"minEc is higher than initial Coulomb"
                            " energy! ("+str(self._minEc)+' > '+str(np.sum(self._Ec))+")")

        self._Q = getQValue(self._fp.mEx,self._pp.mEx,self._tp.mEx,self._hf.mEx,self._lf.mEx,self._lostNeutrons)
                                  
        # Check that Q value is reasonable
        if self._Q < 0:
            _throwException(self,'Exception','Negative Q value (='+str(self._Q)+"). It needs to"
                            " be positive.")
        
        # Check that Coulomb energy is not too great
        if self._Q < np.sum(self._Ec):
            _throwException(self,'Exception',"Energy not conserved: Particles are too close, "
                            "generating a Coulomb Energy > Q ("+str(np.sum(self._Ec))+">"+str(self._Q)+").")
                            
        # Check that total energy is conserved
        if self._Q < (np.sum(self._Ekin) + np.sum(self._Ec)):
            _throwException(self,'Exception',"Energy not conserved: Ekin + Ec > Q ("+\
                                 str(np.sum(self._Ec)+np.sum(self._Ekin))+">"+str(self._Q)+").")
                            
        # Check that r is in proper format
        if len(self._r) != 6:
            _throwException(self,'Exception',"r needs to include 6 initial coordinates, i.e."
                            " x and y for the fission fragments.")
        for i in self._r:
            if not isinstance(i, float) and i != 0:
                _throwException(self,'TypeError','All elements in r must be float, (or atleast int if zero).')
            if not np.isfinite(i) or i == None:
                _throwException(self,'ValueError','All elements in r must be set to a finite value.')
            

        # Check that v is in proper format
        if len(self._v) != 6:
            _throwException(self,'Exception',"v needs to include 6 initial velocities, i.e."
                            " vx and vy for the fission fragments.")
        for i in self._v:
            if not isinstance(i, float) and i != 0:
                _throwException(self,'TypeError','All elements in v must be float, (or atleast int if zero).')
            if not np.isfinite(i) or i == None:
                _throwException(self,'ValueError','All elements in v must be set to a finite value.')
        
        # Check that particles do not overlap
        if getDistance(self._r[0:2],self._r[2:4]) < (self._rtp + self._rhf):
            _throwException(self,'Exception',"TP and HF are overlapping: r=<r_tp+r_hf"
                            " ("+str(getDistance(self._r[0:2],self._r[2:4]))+"<"+str(self._rtp+self._rhf)+"). "
                            "Increase their initial spacing.")
        if getDistance(self._r[0:2],self._r[4:6]) < (self._rtp + self._rlf):
            _throwException(self,'Exception',"TP and LF are overlapping: r=<r_tp+r_lf"
                            " ("+str(getDistance(self._r[0:2],self._r[4:6]))+"<"+str(self._rtp+self._rlf)+"). "
                            "Increase their initial spacing.")
        if getDistance(self._r[2:4],self._r[4:6]) < (self._rhf + self._rlf):
            _throwException(self,'Exception',"HF and LF are overlapping: r=<r_hf+r_lf"
                            " ("+str(getDistance(self._r[2:4],self._r[4:6]))+"<"+str(self._rhf+self._rlf)+"). "
                            "Increase their initial spacing.")
        
        # Assign initial speeds with remaining kinetic energy
        
        # Check that total angular momentum is conserved
    
    def run(self, simulationNumber=1, timeStamp=None):
        """
        Runs simulation by solving the ODE for the initialized system.
        """
        if timeStamp == None:
            timesTamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
        
        def odeFunction(u, dt):
            """
            Function containing the equations of motion.
            
            :type u: list of float
            :param u: List containing positions and velocities.
            
            :type dt: list of float
            :param dt: Time interval to solve the ODE for.
            
            :rtype: list of float
            :returns: List of solved velocities and accelerations for fission
                     fragments.
            """

            r12x = u[0]-u[2]
            r12y = u[1]-u[3]
            r13x = u[0]-u[4]
            r13y = u[1]-u[5]
            r23x = u[2]-u[4]
            r23y = u[3]-u[5]
            d12 = np.sqrt((r12x)**2 + (r12y)**2)
            d13 = np.sqrt((r13x)**2 + (r13y)**2)
            d23 = np.sqrt((r23x)**2 + (r23y)**2)
            k = 1.439964
            c12 = k*(self._Z[0])*(self._Z[1])
            c13 = k*(self._Z[0])*(self._Z[2])
            c23 = k*(self._Z[1])*(self._Z[2])
            
            atpx = c12*r12x / (self._mff[0] * d12**3) + c13*r13x / (self._mff[0] * d13**3)
            atpy = c12*r12y / (self._mff[0] * d12**3) + c13*r13y / (self._mff[0] * d13**3)
            ahx = -c12*r12x / (self._mff[1] * d12**3) + c23*r23x / (self._mff[1] * d23**3)
            ahy = -c12*r12y / (self._mff[1] * d12**3) + c23*r23y / (self._mff[1] * d23**3)
            alx = -c13*r13x / (self._mff[2] * d13**3) - c23*r23x / (self._mff[2] * d23**3)
            aly = -c13*r13y / (self._mff[2] * d13**3) - c23*r23y / (self._mff[2] * d23**3)
            #return (u[6:12] + self._pint.accelerations(self._Z, u[0:6], self._mff))

            return [u[6], u[7], u[8], u[9], u[10], u[11],
                    atpx, atpy, ahx, ahy, alx, aly]


        #if self._verbose and self._exceptionCount == 0:
        #    print("------Starting simulation number "+str(simulationNumber)+" with initial configuration------")
        #    print("Reaction: "+str(self._fp.name)+"+"+str(self._pp.name)+" -> "
        #          ""+str(self._hf.name)+"+"+str(self._lf.name)+"+"+\
        #          str(self._tp.name))
        #    print("Q-Value: "+str(self._Q)+" MeV/c^2")
        #    print("Ec,0: "+str(np.sum(self._Ec))+" MeV/c^2\tStop at Ec="+str(100.0*self._minEc)+"\%")
        #    print("Ekin,0: "+str(np.sum(self._Ekin))+" MeV/c^2")
        #    print("r(tp,hf,lf): "+str(self._r))
        #    print("-------------------------------------------------------------------")
        #    print('Run:0\tEc: '+str(np.sum(self._Ec))+'\tEkin: '+str(np.sum(self._Ekin))+'\tEc+Ekin: '+str(np.sum(self._Ekin)+np.sum(self._Ec)))
            
        runNumber = 0
        startTime = time()
        dt = np.arange(0.0, 1000.0, 0.01)
        self._filePath = "results/" + str(self._simulationName) + "/" + \
                         str(timeStamp) + "/"
        
        # Create results folder if it doesn't exist
        if not os.path.exists(self._filePath):
            os.makedirs(self._filePath)
        
        # Create human readable file with system information
        #with open(self._filePath + "systemInfo.txt", 'ab') as f_info:
        #    f_info.write("---------- System Info ----------\n")
        #    f_info.write("Reaction: "+str(self._fp.name)+"("+ \
        #                 str(self._pp.name)+",f) -> "+str(self._hf.name)+"+"+ \
        #                 str(self._lf.name)+"+"+str(self._tp.name)+"\n")
        #    f_info.write("Q-value: "+str(self._Q)+"\n")
        #    f_info.write("Ec,0: "+str(self._Ec)+"\n")
        #    f_info.write("Ekin,0: "+str(self._Ekin)+"\n")
        #    f_info.write("r0: "+str(self._r)+"\n")
        #    f_info.write("v0: "+str(self._v)+"\n")

        self._v0 = self._v
        self._r0 = self._r
        self._Ec0 = self._Ec
        self._Ekin0 = self._Ekin
        
        
        while (runNumber < self._maxRunsODE or self._maxRunsODE == 0) and \
              ((time()-startTime) < self._maxTimeODE or self._maxTimeODE == 0) and \
              self._exceptionCount == 0 and \
              np.sum(self._Ec) >= self._minEc*np.sum(self._Ec0):
            runTime = time()
            runNumber += 1
            xtp, ytp, xhf, yhf, xlf, ylf, vxtp, vytp, vxhf, vyhf, vxlf, vylf = \
                odeint(odeFunction, (self._r + self._v), dt).T

            self._r = [xtp[-1],ytp[-1],xhf[-1],yhf[-1],xlf[-1],ylf[-1]]
            self._v = [vxtp[-1],vytp[-1],vxhf[-1],vyhf[-1],vxlf[-1],vylf[-1]]

            # Free up some memory
            del vxtp, vytp, vxhf, vyhf, vxlf, vylf
            
            # Check if Coulomb energy is below the set threshold
            self._Ec = self._pint.coulombEnergies(self._Z, self._r)
            
            # Get the current kinetic energy
            self._Ekin = getKineticEnergies(self)

            # Check that none of the particles behave oddly
            if not np.isfinite(np.sum(self._Ec)):
                _throwException(self,'Exception',"Coulomb Energy not finite after run number "+str(runNumber)+"! Ec="+str(np.sum(self._Ec)))
            if not np.isfinite(np.sum(self._Ekin)):
                _throwException(self,'Exception',"Kinetic Energy not finite after run number "+str(runNumber)+"! Ekin="+str(np.sum(self._Ekin)))
            
            # Check that kinetic energy is reasonable compared to Q-value
            if (np.sum(self._Ekin) + np.sum(self._Ec)) > self._Q :
                _throwException(self,'Exception',"Kinetic + Coulomb energy higher than initial "
                                "Q-value. This breaks energy conservation! Run: "+\
                                str(runNumber)+", Ekin: "+str(self._Ekin)+", Ec: "+str(self._Ec))
            
            # Print info about the run to the terminal, if verbose mode is on
            #if self._verbose:
            #    print('Run:'+str(runNumber)+'\tEc: '+str(np.sum(self._Ec))+
            #1          '\tEkin: '+str(np.sum(self._Ekin))+'\tEc+Ekin: '+str(np.sum(self._Ekin)+np.sum(self._Ec))+
            #          '\tTime: '+str(time()-runTime)+'\tAngle: '+str(getAngle(self._r[0:2],self._r[4:6]))+'\ttsteps:'+str(len(xtp)))
            # Save paths to file to free up memory
            if self._saveTrajectories:
                if not self._exceptionCount > 0:
                    f_data = file(self._filePath + "trajectories_"+str(runNumber)+".bin", 'wb')
                    np.save(f_data,np.array([xtp,ytp,xhf,yhf,xlf,ylf]))
                    f_data.close()
            # Free up some memory
            del xtp, ytp, xhf, yhf, xlf, ylf
        
            #if self._exceptionCount == 0:
            #    # Store relevant variables for this run in a shelved file format
            #    s = shelve.open(self._filePath + 'shelvedVariables.sb')
            #    try:
            #        s['run'+str(runNumber)] = { 'angle': getAngle(self._r[0:2],self._r[4:6]), 'Ec': self._Ec, 'Ekin': self._Ekin }
            #    finally:
            #        s.close()
        # end of loop
        stopTime = time()
        #if self._verbose and self._exceptionCount == 0:
        #    print("----------------------------------------------------------")
        #    print("Total run time: "+str(time()-startTime))

        # Store final info in a human readable text file
        #with open(self._filePath + "systemInfo.txt", 'ab') as f_data:
        #    if self._exceptionCount == 0:
        #        # Add final energies if no error occured
        #        f_data.write("Ec,final: "+str(self._Ec)+"\n")
        #        f_data.write("Ekin,final: "+str(self._Ekin)+"\n")
        #        f_data.write("Angle: " +str(getAngle(self._r[0:2],
        #                                             self._r[4:6]))+"\n")
        #        f_data.write("Total run time: "+str(time()-startTime)+"\n")
        #        f_data.write("Amount of runs: "+str(runNumber)+"\n")
        #    else:
        #        # Add exceptionMessage to systemInfo.txt if there was an error
        #        f_data.write("Error: "+str(self._exceptionMessage)+"\n")
                
        # Throw exception if the maxRunsODE was reached before convergence
        if runNumber == self._maxRunsODE and np.sum(self._Ec) > self._minEc*np.sum(self._Ec0):
            _throwException(self,'ExceptionError','Maximum allowed runs (maxRunsODE) was reached before convergence.')
            
        if (stopTime-startTime) >= self._maxTimeODE and np.sum(self._Ec) > self._minEc*np.sum(self._Ec0):
            _throwException(self,'ExceptionError','Maximum allowed runtime (maxTimeODE) was reached before convergence.')
        
        # Store variables and their final values in a shelved file format
        if self._exceptionCount > 0:
            shelveStatus = 1
            shelveError = self._exceptionMessage
        else:
            shelveStatus = 0
            shelveError = None
            
        # Store variables and their final values in a shelved file format
        s = shelve.open(self._filePath + 'shelvedVariables.sb')
        try:
            s[str(simulationNumber)] = {'simName': self._simulationName,
                                        'fissionType': self._fissionType,
                                        #'particles': [self._particles[0].name, 
                                        #              self._particles[1].name, 
                                        #              self._particles[2].name]
                                        'Q': self._Q,
                                        'r0': self._r0,
                                        'v0': self._v0,
                                        'Ec0': self._Ec0,
                                        'Ekin0': self._Ekin0,
                                        'angle': getAngle(self._r[0:2],self._r[4:6]),
                                        'Ec': self._Ec,
                                        'Ekin': self._Ekin,
                                        'runs': runNumber,
                                        'status': shelveStatus,
                                        'error': shelveError}
        finally:
            s.close()
    
        if self._exceptionCount == 0:
            outString = "a:"+str(getAngle(self._r[0:2],self._r[4:6]))+"\tEi:"+str(np.sum(self._Ec0)+np.sum(self._Ekin0))
        else:
            outString = ""
        return self._exceptionCount, outString
    # end of run()
    def plotTrajectories(self):
        """
        Plot trajectories.
        """

        with open(self._filePath + "trajectories_1.bin","rb") as f_data:
            r = np.load(f_data)
            
            pl.figure(1)
            pl.plot(r[0],r[1],'r-')
            pl.plot(r[2],r[3],'g-')
            pl.plot(r[4],r[5],'b-')    
            pl.show()

    def animateTrajectories(self):
        """
        Animate trajectories.
        """
        with open(self._filePath + "trajectories_1.bin","rb") as f_data:
            r = np.load(f_data)
        plt.ion()
        plt.axis([np.floor(np.amin([r[0,],r[2],r[4]])),
                  np.ceil(np.amax([r[0],r[2],r[4]])),
                  np.floor(np.amin([r[1],r[3],r[5]])),
                  np.amax([r[1],r[3],r[5]])])
        plt.show()
        
        for i in range(0,len(r[0])):
            plt.clf()
            plt.axis([np.floor(np.amin([r[0],r[2],r[4]])),
                      np.ceil(np.amax([r[0],r[2],r[4]])),
                      np.floor(np.amin([r[1],r[3],r[5]])),
                      np.amax([r[1],r[3],r[5]])])
            plt.scatter(r[0][i],r[1][i],c='r',s=np.int(self._mff[0]/100.0))
            plt.scatter(r[2][i],r[3][i],c='g',s=np.int(self._mff[1]/100.0))
            plt.scatter(r[4][i],r[5][i],c='b',s=np.int(self._mff[2]/100.0))
            plt.plot(r[0][0:i],r[1][0:i],'r-',lw=2.0)
            plt.plot(r[2][0:i],r[3][0:i],'g:',lw=4.0)
            plt.plot(r[4][0:i],r[5][0:i],'b--',lw=2.0)
            
            plt.draw()
            sleep(0.01)
        plt.show()

    def getFilePath(self):
        """
        Get the file path, used by the plotter for example.
        
        :rtype: string
        :returns: File path to trajectories/systemInfo.
        """
        return self._filePath

def _throwException(self, exceptionType_in, exceptionMessage_in):
    """
    Wrapper for throwing exceptions, in order to make it possible to switch
    between interrupting the program or letting it continue on with another
    simulation.
    
    :type exceptionType_in: 
    :param exceptionType_in: What exception to throw.
    
    :type exceptionMessage_in: string
    :param exceptionMessage_in: Message to show in exception/simulation status.
    """
    if self._interruptOnException:
        raise Exception(str(exceptionMessage_in))
    else:
        if self._exceptionCount == 0:
            print(str(exceptionType_in)+': '+str(exceptionMessage_in))
            self._exceptionMessage = exceptionMessage_in
            self._exceptionCount = 1
        else:
            self._exceptionCount += 1

def getKineticEnergies(self):
    """
    Retruns kinetic energies of the particles.

    :rtype: list of floats
    :returns: A list of the kinetic energies (E=m*v^2/2):
              [E_tp, E_hf, E_lf]
    """
    return [self._mff[0]*(self._v[0]**2+self._v[1]**2)*0.5,
            self._mff[1]*(self._v[2]**2+self._v[3]**2)*0.5,
            self._mff[2]*(self._v[4]**2+self._v[5]**2)*0.5]


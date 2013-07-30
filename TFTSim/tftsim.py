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
import commands
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

    def __init__(self, sa, r_in, v_in, TXE=0.0):
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

        self._r = r_in
        self._v = v_in
        self._simulationName = sa.simulationName
        self._fissionType = sa.fissionType
        self._cint = copy.copy(sa.cint)
        self._nint = copy.copy(sa.nint)
        self._fp = copy.copy(sa.fp)
        self._pp = copy.copy(sa.pp)
        if self._fissionType != 'BF':
            self._tp = copy.copy(sa.tp)
        self._hf = copy.copy(sa.hf)
        self._lf = copy.copy(sa.lf)
        self._lostNeutrons = sa.lostNeutrons
        self._betas = sa.betas
        self._minEc = sa.minEc
        self._maxRunsODE = sa.maxRunsODE
        self._maxTimeODE = sa.maxTimeODE
        self._neutronEvaporation = sa.neutronEvaporation
        self._verbose = sa.verbose
        self._interruptOnException = sa.interruptOnException
        self._collisionCheck = sa.collisionCheck
        self._saveTrajectories = sa.saveTrajectories
        self._saveKineticEnergies = sa.saveKineticEnergies
        self._useGPU = sa.useGPU
        self._GPU64bitFloat = sa.GPU64bitFloat
        
        self._TXE = TXE
        self._mff = sa.mff
        self._Z = sa.Z
        self._rad = sa.rad
        self._ab = sa.ab
        self._ec = sa.ec
        self._Q = sa.Q
   
        self._exceptionCount = 0
        self._exceptionMessage = None
        
        # Check that simulationName is a valid string
        if not isinstance(self._simulationName, basestring):
            _throwException(self,TypeError, 'simulationName must be a string.')
        if self._simulationName == None or self._simulationName == '':
            _throwException(self,ValueError, 'simulationName must be set to a non-empty string.')
            
        # Check that fissionType is a valid string
        if not isinstance(self._fissionType, basestring):
            _throwException(self,TypeError, 'fissionType must be a string.')
        if self._fissionType == None or self._fissionType == '':
            _throwException(self,ValueError, 'fissionType must be set to a non-empty string.')
        else:
            if self._fissionType not in ['LCP','CCT','BF']:
                _throwException(self,ValueError,'Invalid fissionType, must be one of LCP, CCT, BF.')
        
        # Check that lost neutron number is in proper format
        if not isinstance(self._lostNeutrons, int):
            _throwException(self,TypeError, 'lostNeutrons must be an integer.')
        if self._lostNeutrons == None or self._lostNeutrons < 0:
            _throwException(self,Exception,'lostNeutrons must be set to a value >= 0.')

        # Check that particle number is conserved
        if self._pp != None:
            in_part_num = self._fp.A + self._pp.A
        else:
            in_part_num = self._fp.A
        
        if self._fissionType == 'BF':
            out_part_num = self._lostNeutrons + self._hf.A + self._lf.A
        else:
            out_part_num = self._lostNeutrons + self._tp.A + self._hf.A + self._lf.A
        if out_part_num != in_part_num:
            _throwException(self,Exception,"Nucleon number not conserved! A_in "
                                           " = "+str(in_part_num)+", A_out = "+\
                                           str(out_part_num))
        
        # Check that particles are correctly ordered after increasing size
        if self._fissionType == 'LCP':
            if self._tp.A > self._hf.A:
                _throwException(self,Exception,"Ternary particle is heavier than the heavy fission"
                                " fragment! ("+str(self._tp.A)+">"+str(self._hf.A)+")")
            if self._tp.A > self._lf.A:
                _throwException(self,Exception,"Ternary particle is heavier than the light fission"
                                " fragment! ("+str(self._tp.A)+">"+str(self._lf.A)+")")
            if self._lf.A > self._hf.A:
                _throwException(self,Exception,"Light fission fragment is heavier than the heavy "
                                "fission fragment! ("+str(self._lf.A)+">"+str(self._hf.A)+")")

        # Check that minEc is in proper format
        if not isinstance(self._minEc, float):
            _throwException(self,TypeError,'minEc needs to be a float.')
        if self._minEc == None or self._minEc <= 0 or self._minEc >= 1:
            _throwException(self,Exception,'minEc must be set to a value 0 < minEc < 1.')

        # Check that maxRunsODE is in proper format
        if not isinstance(self._maxRunsODE, int):
            _throwException(self,TypeError,'maxRunsODE needs to be an int.')
        if self._maxRunsODE == None or self._maxRunsODE < 0 or not np.isfinite(self._maxRunsODE):
            _throwException(self,Exception,"maxRunsODE must be set to a finite value >= 0."
                                             "(0 means indefinite runs until convergence)")
        # Check that maxTimeODE is in proper format
        if not isinstance(self._maxTimeODE, int) and not isinstance(self._maxTimeODE, float):
            _throwException(self,TypeError,'maxRunsODE needs to be float or int.')
        if self._maxRunsODE == None or self._maxRunsODE < 0 or not np.isfinite(self._maxTimeODE):
            _throwException(self,Exception,"maxRunsODE must be set to a finite value >= 0."
                                             "(0 means indefinite run time until until convergence)")

        self._Ec = self._cint.coulombEnergies(self._Z, self._r,fissionType_in=self._fissionType)

        self._Ekin = getKineticEnergies(self)
        
        
        # Check that Ec is a number

        # Check that minEc is not too high
        if self._minEc >= np.sum(self._Ec):
            _throwException(self,Exception,"minEc is higher than initial"
                                           "Coulomb energy! ("+\
                                           str(self._minEc)+' > '+\
                                           str(np.sum(self._Ec))+")")
                                  
        # Check that Q value is reasonable
        if self._Q < 0:
            _throwException(self,Exception,"Negative Q value (="+str(self._Q)+\
                                           "). It needs to be positive.")
        
        # Check that Coulomb energy is not too great
        if self._Q < np.sum(self._Ec):
            _throwException(self,Exception,"Energy not conserved: Particles are"
                                           " too close, generating a Coulomb "
                                           "Energy > Q ("+\
                                           str(np.sum(self._Ec))+">"+\
                                           str(self._Q)+"). Ec="+str(self._Ec))
                            
        # Check that total energy is conserved
        if self._Q < (np.sum(self._Ekin) + np.sum(self._Ec)):
            _throwException(self,Exception,"Energy not conserved: TXE + Ekin + "
                                           "Ec > Q ("+str(np.sum(self._Ec)+\
                                           self._TXE+np.sum(self._Ekin))+">"+\
                                           str(self._Q)+")")
                            
        # Check that r is in proper format
        if self._fissionType == 'BF':
            if len(self._r) != 4:
                _throwException(self,Exception,"r needs to include 4 initial "
                                               "coordinates, i.e. x and y for "
                                               "the fission fragments.")
        else:
            if len(self._r) != 6:
                _throwException(self,Exception,"r needs to include 6 initial "
                                               "coordinates, i.e. x and y for "
                                               "the fission fragments.")
        for i in self._r:
            if not isinstance(i, float) and i != 0:
                _throwException(self,TypeError,"All elements in r must be float"
                                               ", (or atleast int if zero).")
            if not np.isfinite(i) or i == None:
                _throwException(self,ValueError,"All elements in r must be set "
                                                "to a finite value.")
            

        # Check that v is in proper format
        if self._fissionType == 'BF':
            if len(self._v) != 4:
                _throwException(self,Exception,"v needs to include 4 initial "
                                               "velocities, i.e. vx and vy for the "
                                               "fission fragments.")        
        else:
            if len(self._v) != 6:
                _throwException(self,Exception,"v needs to include 6 initial "
                                               "velocities, i.e. vx and vy for the "
                                               "fission fragments.")
        for i in self._v:
            if not isinstance(i, float) and i != 0:
                _throwException(self,TypeError,"All elements in v must be float"
                                               ", (or atleast int if zero).")
            if not np.isfinite(i) or i == None:
                _throwException(self,ValueError,"All elements in v must be set "
                                                "to a finite value.")
        
        
        # Check that particles do not overlap
        if self._fissionType == 'BF':
            if abs(self._r[0]-self._r[2]) <= (self._ab[0] + self._ab[2]):
                _throwException(self,ValueError,"HF and LF tip distance is less than 1 fm: "
                                " ("+str(abs(self._r[0]-self._r[2]))+\
                                " <= "+str(self._ab[0] + self._ab[2])+"). "
                                "Increase their initial spacing.")      
        else:
            if circleEllipseOverlap(self._r[0:4], self._ab[2], self._ab[3], self._rad[0]):
                _throwException(self,ValueError,"TP and HF are overlapping: "
                                " ("+str((self._r[2]-self._r[0])**2/(self._ab[2]+self._rad[0])**2 + \
                                         (self._r[3]-self._r[1])**2/(self._ab[3]+self._rad[0])**2)+" <= 1). "
                                "Increase their initial spacing.")
            if circleEllipseOverlap(self._r[0:2]+self._r[4:6], self._ab[4], self._ab[5], self._rad[0]):
                _throwException(self,ValueError,"TP and LF are overlapping: "
                                " ("+str((self._r[4]-self._r[0])**2/(self._ab[4]+self._rad[0])**2 + \
                                         (self._r[5]-self._r[1])**2/(self._ab[5]+self._rad[0])**2)+" <= 1). "
                                "Increase their initial spacing.")
            if abs(self._r[2]-self._r[4]) <= (self._ab[2] + self._ab[4]):
                _throwException(self,ValueError,"HF and LF tip distance is less than 1 fm: "
                                " ("+str(abs(self._r[2]-self._r[4]))+\
                                " <= "+str(self._ab[2] + self._ab[4])+"). "
                                "Increase their initial spacing.")  

        # Check that total linear momentum is conserved
        # Check that total angular momentum is conserved
    
    def run(self, simulationNumber=1, timeStamp=None):
        """
        Runs simulation by solving the ODE for equations of motions on the CPU
        for the initialized system.
        """
        if timeStamp == None:
            timeStamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
        
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
            
            a_out = self._cint.accelerations(Z_in = self._Z,
                                             r_in = list(u[0:int(len(u)/2)]),
                                             m_in = self._mff,
                                             fissionType_in=self._fissionType)
            
            return list(u[int(len(u)/2):len(u)]) + a_out

        runNumber = 0
        startTime = time()
        dt = np.arange(0.0, 1000.0, 0.01)
        self._filePath = "results/" + str(self._simulationName) + "/" + \
                         str(timeStamp) + "/"
        
        # Create results folder if it doesn't exist
        if not os.path.exists(self._filePath):
            os.makedirs(self._filePath)
        
        self._v0 = self._v
        self._r0 = self._r
        self._Ec0 = self._Ec
        self._Ekin0 = self._Ekin
        
        if self._fissionType == 'BF':
            self._D = abs(self._r0[2])
        else:
            self._D = abs(self._r0[4]-self._r0[2])
        
        ekins = [] # Used to store Kinetic energy after each ODErun
        if self._saveKineticEnergies:
            ekins = [np.sum(self._Ekin)]
        
        while (runNumber < self._maxRunsODE or self._maxRunsODE == 0) and \
              ((time()-startTime) < self._maxTimeODE or self._maxTimeODE == 0) and \
              self._exceptionCount == 0 and \
              np.sum(self._Ec) >= self._minEc*np.sum(self._Ec0):
            runTime = time()
            runNumber += 1
            
            
            #xtp, ytp, xhf, yhf, xlf, ylf, vxtp, vytp, vxhf, vyhf, vxlf, vylf = \
            #    odeint(odeFunction, (self._r + self._v), dt).T
            ode_sol = odeint(odeFunction, (self._r + self._v), dt).T
            
            if self._collisionCheck:
                for i in range(0,len(ode_sol[0,:])):
                    if self._fissionType != 'BF' and \
                       circleEllipseOverlap(r_in=[ode_sol[0,i],ode_sol[1,i],
                                                  ode_sol[2,i],ode_sol[3,i]],
                                            a_in=self._ab[2], b_in=self._ab[3],
                                            rad_in=self._rad[0]):
                        _throwException(self,Exception,"TP and HF collided "
                                                       "during acceleration!")
                    if self._fissionType != 'BF' and \
                       circleEllipseOverlap(r_in=[ode_sol[0,i],ode_sol[1,i],
                                                  ode_sol[4,i],ode_sol[5,i]],
                                            a_in=self._ab[4], b_in=self._ab[5],
                                            rad_in=self._rad[0]):
                        _throwException(self,Exception,"TP and LF collided "
                                                       "during acceleration!")
                    if (ode_sol[-2-int(len(ode_sol[:,0])/2),i]-\
                        ode_sol[-4-int(len(ode_sol[:,0])/2),i]) < \
                                                    (self._ab[-2]+self._ab[-4]):
                        _throwException(self,Exception,"HF and LF collided "
                                                       "during acceleration!")
            
            self._r = list(ode_sol[0:int(len(ode_sol)/2),-1])
            self._v = list(ode_sol[int(len(ode_sol)/2):len(ode_sol),-1])
            #self._r = [ode_sol[0][-1],ode_sol[1][-1],ode_sol[2][-1],ode_sol[3][-1],ode_sol[4][-1],ode_sol[5][-1]]
            #self._v = [ode_sol[6][-1],ode_sol[7][-1],ode_sol[8][-1],ode_sol[9][-1],ode_sol[10][-1],ode_sol[11][-1]]
            #self._r = [xtp[-1],ytp[-1],xhf[-1],yhf[-1],xlf[-1],ylf[-1]]
            #self._v = [vxtp[-1],vytp[-1],vxhf[-1],vyhf[-1],vxlf[-1],vylf[-1]]
            
            x_cm, y_cm = getCentreOfMass(r_in=self._r, m_in=self._mff)


            # Free up some memory
            #del vxtp, vytp, vxhf, vyhf, vxlf, vylf
            
            # Check if Coulomb energy is below the set threshold
            self._Ec = self._cint.coulombEnergies(Z_in=self._Z, r_in=self._r,fissionType_in=self._fissionType)
            
            # Get the current kinetic energy
            self._Ekin = getKineticEnergies(self)
            if self._saveKineticEnergies:
                ekins.append(np.sum(self._Ekin))

            # Check that none of the particles behave oddly
            if not np.isfinite(np.sum(self._Ec)):
                _throwException(self,Exception,"Coulomb Energy not finite after"
                                               " run number "+str(runNumber)+\
                                               "! Ec="+str(np.sum(self._Ec)))
            if not np.isfinite(np.sum(self._Ekin)):
                _throwException(self,Exception,"Kinetic Energy not finite after"
                                                " run number "+str(runNumber)+\
                                                "! Ekin="+\
                                                str(np.sum(self._Ekin)))
            
            # Check that kinetic energy is reasonable compared to Q-value
            if (self._TXE + np.sum(self._Ekin) + np.sum(self._Ec)) > self._Q :
                _throwException(self,Exception,"Excitation energy + "
                                               "Kinetic + Coulomb energy higher"
                                               " than initial Q-value. This "
                                               "breaks energy conservation! "
                                               "Run: "+str(runNumber)+", "+\
                                               str(np.sum(self._Ekin)+\
                                                   self._TXE+\
                                                   np.sum(self._Ec))+">"+\
                                               str(self._Q)+"\tEc: "+\
                                               str(self._Ec)+" Ekin: "+\
                                               str(self._Ekin)+" TXE:"+\
                                               str(self._TXE))
            
            # Save paths to file to free up memory
            if self._saveTrajectories:
                if not self._exceptionCount > 0:
                    f_data = file(self._filePath + "trajectories_"+\
                                  str(runNumber)+".bin", 'wb')
                    np.save(f_data,np.array([ode_sol[0],ode_sol[1],ode_sol[2],ode_sol[3],ode_sol[4],ode_sol[5],
                                             np.ones(len(ode_sol[0]))*x_cm,
                                             np.ones(len(ode_sol[0]))*y_cm]))
                    #np.save(f_data,np.array([xtp,ytp,xhf,yhf,xlf,ylf,
                    #                         np.ones(len(xtp))*x_cm,
                    #                         np.ones(len(xtp))*y_cm]))
                    f_data.close()
            # Free up some memory
            del ode_sol
            #del xtp, ytp, xhf, yhf, xlf, ylf
        
            
        # end of while-loop
        stopTime = time()
                
        # Throw exception if the maxRunsODE was reached before convergence
        
        if runNumber == self._maxRunsODE and np.sum(self._Ec) > self._minEc*np.sum(self._Ec0):
            _throwException(self,'ExceptionError',"Maximum allowed runs "
                                                  "(maxRunsODE) was reached "
                                                  "before convergence.")
            
        if (stopTime-startTime) >= self._maxTimeODE and np.sum(self._Ec) > self._minEc*np.sum(self._Ec0):
            _throwException(self,'ExceptionError',"Maximum allowed runtime "
                                                  "(maxTimeODE) was reached "
                                                  "before convergence.")
        
        # Store variables and their final values in a shelved file format
        if self._exceptionCount > 0:
            shelveStatus = 1
            shelveError = self._exceptionMessage
        else:
            shelveStatus = 0
            shelveError = None
        
        # Mirror the configuration if the tp goes "through" to the other side
        if self._fissionType != 'BF' and self._r[1] < 0:
            self._r[1] = -self._r[1]
            self._r[3] = -self._r[3]
            self._r[5] = -self._r[5]
            self._v[1] = -self._v[1]
            self._v[3] = -self._v[3]
            self._v[5] = -self._v[5]
            wentThrough = True
        else:
            wentThrough = False
            
        
        # Store variables and their final values in a shelved file format
        s = shelve.open(self._filePath + 'shelvedVariables.sb')
        try:
            s[str(simulationNumber)] = {'simName': self._simulationName,
                                        'simNumber': simulationNumber,
                                        'fissionType': self._fissionType,
                                        'Q': self._Q,
                                        'D': self._D,
                                        'r': self._r,
                                        'v': self._v,
                                        'r0': self._r0,
                                        'v0': self._v0,
                                        'TXE': self._TXE,
                                        'Ec0': self._Ec0,
                                        'Ekin0': self._Ekin0,
                                        'angle': getAngle(self._r[0:2],
                                                          self._r[-2:len(self._r)]),
                                        'Ec': self._Ec,
                                        'Ekin': self._Ekin,
                                        'ODEruns': runNumber,
                                        'status': shelveStatus,
                                        'error': shelveError,
                                        'time': stopTime-startTime,
                                        'wentThrough': wentThrough,
                                        'Ekins': ekins}
        finally:
            s.close()
        
        if simulationNumber == 1:
            # Store static variables in a shelved file format
            s = shelve.open(self._filePath + 'shelvedStaticVariables.sb')
            if self._fissionType == 'BF':
                particles = []
            else:
                particles = [self._tp]
            particles.extend([self._hf,self._lf])
            try:
                s[str(simulationNumber)] = {'simName': self._simulationName,
                                            'fissionType': self._fissionType,
                                            'particles': particles,
                                            'coulombInteraction': self._cint,
                                            'nuclearInteraction': self._nint,
                                            'Q': self._Q,
                                            'D': self._D, # Note that this might not be a static variable
                                            'ab': self._ab,
                                            'ec': self._ec
                                            }
            finally:
                s.close()
            
        
        if self._exceptionCount == 0:
            outString = "a:"+str(getAngle(self._r[0:2],
                                          self._r[-2:len(self._r)]))+"\tEi:"+\
                        str(np.sum(self._Ec0)+np.sum(self._Ekin0))
        else:
            outString = ""
        return self._exceptionCount, outString, self._Ekin[0]
    # end of runCPU()
    
    
    def runGPU(self, simulations, r0, v0):
        # Import PyOpenCL and related sub packages
        import pyopencl as cl
        import pyopencl.array
        import pyopencl.clrandom
        import pyopencl.clmath
        
        # Preprocessor defines for the compiler of the OpenCL-code
        defines = ""
        if self._GPU64bitFloat:
            defines += "#define ENABLE_DOUBLE\n"
        if self._fissionType == 'BF':
            defines += "#define BINARY_FISSION\n"
        if self._collisionCheck:
            defines += "#define COLLISION_CHECK\n"
        
        class DictWithDefault(defaultdict):
            def __missing__(self, key):
                return key + str(" is not defined")
                
        # Constants used in the kernel code will be pasted into the kernel code
        # through the replacements dictionary.
        replacements = DictWithDefault()
        replacements['dt'] = '%f' % self._dt
        replacements['defines'] = defines
        replacements['Q'] = '%f' self._Q
        #ec, z, m, rad, ab
        replacements['ec1'] = '%f' self._ec[0]
        replacements['ec2'] = '%f' self._ec[1]
        replacements['ec3'] = '%f' self._ec[2]
        replacements['ab1'] = '%f' self._ab[0]
        replacements['ab2'] = '%f' self._ab[1]
        replacements['ab3'] = '%f' self._ab[2]
        replacements['Z1'] = '%d' self._Z[0]
        replacements['Z2'] = '%d' self._Z[1]
        replacements['Z3'] = '%d' self._Z[2]
        replacements['m1'] = '%f' self._mff[0]
        replacements['m2'] = '%f' self._mff[1]
        replacements['m3'] = '%f' self._mff[2]
        replacements['rad1'] = '%f' self._rad[0]
        replacements['rad2'] = '%f' self._rad[1]
        replacements['rad3'] = '%f' self._rad[2]
        
        
        # Define local and global size of the ND-range
        self._localSize = None
        self._globalSize = (simulations,)
        self._nbrOfThreads = simulations
        
        # Kernel code, paste in constants and defines
        kernelCode_r = open(os.path.dirname(__file__) +
                '/tftsim_kernel.c', 'r').read()
        kernelCode = kernelCode_r % replacements
        
        # Save cleaned and preprocessed results of kernel code
        with open('cleaned.c','w') as f:
            f.write(kernelCode)
        preprocessedCode = commands.getstatusoutput('echo "' +
                kernelCode + '" | cpp')[1]
        cleanedPreprocessedCode = ""
        for i in preprocessedCode.splitlines():
            if len(i) > 0:
                if i[0] != '#':
                    cleanedPreprocessedCode += i + '\n'
        with open('preprocessed.c','w') as f:
            f.write(cleanedPreprocessedCode)
        
        #Create the OpenCL context and command queue
        self._ctx = cl.create_some_context()
        queueProperties = cl.command_queue_properties.PROFILING_ENABLE
        self._queue = cl.CommandQueue(self._ctx,
                                           properties=queueProperties)
        
        programBuildOptions = "-cl-fast-relaxed-math -cl-mad-enable"
        
        #if verbose:
        #    programBuildOptions += " -cl-nv-verbose"
        if not self._enableDouble:
            programBuildOptions += " -cl-single-precision-constant"
        
        #Build the program and identify metropolis as the kernel
        self._prg = (cl.Program(self._ctx, kernelCode)
                         .build(options=programBuildOptions))
        self._kernel = self._prg.gpuODEsolver
        
        
        # Allocate memory for coordinates and velocities on GPU
        try:
            # Coordinates
            self._r_gpu = cl.array.to_device(self._queue,
                   r0.astype(np.float64 if self._GPU64bitFloat else np.float32))
            # Velocities
            self._v_gpu = cl.array.to_device(self._queue,
                   v0.astype(np.float64 if self._GPU64bitFloat else np.float32))
            # Status of simulation
            self._status_gpu = cl.array.zeros(self._queue,
                                              (self._nbrOfThreads, ),
                                              np.uint32)
            # Estimated ODE error size
            self._errorSize_gpu = cl.array.zeros(self._queue,
                                                 (self._nbrOfThreads, 12),
                                                 np.float32)
        except pyopencl.MemoryError:
            raise Exception("Unable to allocate global memory on device,"
                            " out of memory?")
        
        # 
        startTimeGPU = time()
        
        args = [self._r_gpu.data,
                self._v_gpu.data,
                self._status_gpu.data,
                self._errorSize_gpu.data]
        self._kernelObj = self._kernel(self._queue,
                                      self._globalSize,
                                      self._localSize,
                                      *args)
        
        #Wait until the threads have finished and then calculate total run time
        try:
            self._kernelObj.wait()
        except pyopencl.RuntimeError:
            if time() - startTimeGPU > 5.0:
                raise Exception("Kernel runtime error. Over 5 seconds had "
                                "passed when kernel aborted.")
            else:
                raise
        
        # Fetch results from GPU
        r_out = self._r_gpu.get()
        v_out = self._r_gpu.get()        
        status_out = self._status.get()
        
        # Save results in a shelved format
        
        
    # end of runGPU()

    def plotTrajectories(self):
        """
        Plot trajectories.
        """
    
        with open(self._filePath + "trajectories_1.bin","rb") as f_data:
            r = np.load(f_data)
            
            pl.figure(1)
            for i in range(0,int(len(r[:,0])/2)-1):
                pl.plot(r[i*2],r[i*2+1])
                for j in range(0,10):
                    plt.scatter(r[i*2][j*int(len(r[i*2])/10)],r[i*2+1][j*int(len(r[i*2+1])/10)],marker='|',s=40,c='k')
                plt.scatter(r[i*2][-1],r[i*2+1][-1],marker='|',s=40,c='k')
            pl.plot(r[-2],r[-1])
            #pl.plot(r[0],r[1],'r-')
            #pl.plot(r[2],r[3],'g-')
            #pl.plot(r[4],r[5],'b-')
            #pl.plot(r[6],r[7],'k-')
            
            tdist = [0]*int(len(r[:,0])/2)
            for i in range(1,len(r[0])):
                for j in range(0,int(len(r[:,0])/2)):
                    tdist[j] += np.sqrt((r[j*2,i]-r[j*2,i-1])**2 + (r[j*2+1,i]-r[j*2+1,i-1])**2)
            
            
                #t1 += np.sqrt((r[0,i]-r[0,i-1])**2 + (r[1,i]-r[1,i-1])**2)
                #t2 += np.sqrt((r[2,i]-r[2,i-1])**2 + (r[3,i]-r[3,i-1])**2)
                #t3 += np.sqrt((r[4,i]-r[4,i-1])**2 + (r[5,i]-r[5,i-1])**2)
                #t4 += np.sqrt((r[6,i]-r[6,i-1])**2 + (r[7,i]-r[7,i-1])**2)
            #plotEllipse(r[-2,0],r[-1,0],1,1)
            """print('Travel distances: ')
            for t in tdist:
                print(str(t)+' fm')
            #print('TP: '+str(t1)+' fm')
            #print('HF: '+str(t2)+' fm')
            #print('LF: '+str(t3)+' fm')
            #print('CM: '+str(t4)+' fm')
            pl.show()"""


    def animateTrajectories(self):
        """
        Animate trajectories.
        """
        
        with open(self._filePath + "trajectories_1.bin","rb") as f_data:
            r = np.load(f_data)
        plt.ion()
        maxrad = max(self._ab)
        plt.axis([np.floor(np.amin([r[0,],r[2],r[4]]))-maxrad,
                  np.ceil(np.amax([r[0],r[2],r[4]]))+maxrad,
                  min(np.floor(np.amin([r[1],r[3],r[5]])),-maxrad)-maxrad,
                  max(np.amax([r[1],r[3],r[5]]),maxrad)+maxrad])
        
        skipsize = 5000
        for i in range(0,int(len(r[0])/skipsize)):
            plt.clf()
            plt.axis([np.floor(np.amin([r[0,],r[2],r[4]]))-maxrad,
                      np.ceil(np.amax([r[0],r[2],r[4]]))+maxrad,
                      min(np.floor(np.amin([r[1],r[3],r[5]])),-maxrad)-maxrad,
                      max(np.amax([r[1],r[3],r[5]]),maxrad)+maxrad])
            plotEllipse(r[0][i*skipsize],r[1][i*skipsize],self._ab[0],self._ab[1])
            plotEllipse(r[2][i*skipsize],r[3][i*skipsize],self._ab[2],self._ab[3])
            plotEllipse(r[4][i*skipsize],r[5][i*skipsize],self._ab[4],self._ab[5])
            plt.plot(r[0][0:i*skipsize],r[1][0:i*skipsize],'r-',lw=2.0)
            plt.plot(r[2][0:i*skipsize],r[3][0:i*skipsize],'g:',lw=4.0)
            plt.plot(r[4][0:i*skipsize],r[5][0:i*skipsize],'b--',lw=2.0)
            
            plt.draw()
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
    
    :type exceptionType_in: exception
    :param exceptionType_in: Exception type to throw.
    
    :type exceptionMessage_in: string
    :param exceptionMessage_in: Message to show in exception/simulation status.
    """
    if self._interruptOnException:
        raise exceptionType_in(str(exceptionMessage_in))
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
    :returns: A list of the kinetic energies (E=m*v^2/2).
    """
    ekin_out = []
    for i in range(0,len(self._mff)):
        ekin_out.append(self._mff[i]*(self._v[i*2]**2+self._v[i*2+1]**2)*0.5)
    return ekin_out


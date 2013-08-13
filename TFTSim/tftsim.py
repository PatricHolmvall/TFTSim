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
import pickle
import shelve
import pylab as pl
import matplotlib.pyplot as plt
import math

class SimulateTrajectory:
    """
    Simulate the post-scission particle trajectories for a fissioning system.
    """

    def __init__(self, sa):
        """
        Initialize a simulation.
        
        :type sa: :class:`tftsim_args.TFTSimArgs` class
        :param sa: An instance of TFTSimArgs describing what kind of system to
                   simulate.
        """

        self._sa = copy.copy(sa)
        
        """
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
        self._plotInitialConfigs = sa.plotInitialConfigs
        self._displayGeneratorErrors = sa.displayGeneratorErrors
        self._collisionCheck = sa.collisionCheck
        self._saveTrajectories = sa.saveTrajectories
        self._saveKineticEnergies = sa.saveKineticEnergies
        self._useGPU = sa.useGPU
        self._GPU64bitFloat = sa.GPU64bitFloat
        
        self._mff = sa.mff
        self._Z = sa.Z
        self._rad = sa.rad
        self._ab = sa.ab
        self._ec = sa.ec
        self._Q = sa.Q
        """
        
        self._ke2 = 1.43996518
        self._dt = 0.01
        self._odeSteps = 300000
    
        if self._sa.collisionCheck:
            print("Warning! Collision check is turned on. This will slow down "
                  "the speed of simulations greatly.")
    
        # Check that simulationName is a valid string
        if not isinstance(self._sa.simulationName, basestring):
            raise TypeError('simulationName must be a string.')
        if self._sa.simulationName == None or self._sa.simulationName == '':
            raise ValueError('simulationName must be set to a non-empty string.')
            
        # Check that fissionType is a valid string
        if not isinstance(self._sa.fissionType, basestring):
            raise TypeError('fissionType must be a string.')
        if self._sa.fissionType == None or self._sa.fissionType == '':
            raise ValueError('fissionType must be set to a non-empty string.')
        else:
            if self._sa.fissionType not in ['LCP','CCT','BF']:
                raise ValueError('Invalid fissionType, must be one of LCP, CCT, BF.')
        
        # Check that lost neutron number is in proper format
        if not isinstance(self._sa.lostNeutrons, int):
            raise TypeError('lostNeutrons must be an integer.')
        if self._sa.lostNeutrons == None or self._sa.lostNeutrons < 0:
            raise Exception('lostNeutrons must be set to a value >= 0.')
            
        # Check that particle number is conserved
        if self._sa.pp != None:
            in_part_num = self._sa.fp.A + self._sa.pp.A
        else:
            in_part_num = self._sa.fp.A
        
        if self._sa.fissionType == 'BF':
            out_part_num = self._sa.lostNeutrons + self._sa.hf.A + self._sa.lf.A
        else:
            out_part_num = self._sa.lostNeutrons + self._sa.tp.A + self._sa.hf.A + self._sa.lf.A
        if out_part_num != in_part_num:
            raise Exception("Nucleon number not conserved! A_in  = "+\
                            str(in_part_num)+", A_out = "+str(out_part_num))
        
        # Check that particles are correctly ordered after increasing size
        if self._sa.fissionType == 'LCP':
            if self._sa.tp.A > self._sa.hf.A:
                raise Exception("Ternary particle is heavier than the heavy fission"
                                " fragment! ("+str(self._sa.tp.A)+">"+str(self._sa.hf.A)+")")
            if self._sa.tp.A > self._sa.lf.A:
                raise Exception("Ternary particle is heavier than the light fission"
                                " fragment! ("+str(self._sa.tp.A)+">"+str(self._sa.lf.A)+")")
            if self._sa.lf.A > self._sa.hf.A:
                raise Exception("Light fission fragment is heavier than the heavy "
                                "fission fragment! ("+str(self._sa.lf.A)+">"+str(self._sa.hf.A)+")")
        
        # Check that minEc is in proper format
        if not isinstance(self._sa.minEc, float):
            raise TypeError('minEc needs to be a float.')
        if self._sa.minEc == None or self._sa.minEc <= 0 or self._sa.minEc >= 1:
            raise Exception('minEc must be set to a value 0 < minEc < 1.')

        # Check that maxRunsODE is in proper format
        if not isinstance(self._sa.maxRunsODE, int):
            raise TypeError('maxRunsODE needs to be an int.')
        if self._sa.maxRunsODE == None or self._sa.maxRunsODE < 0 or not np.isfinite(self._sa.maxRunsODE):
            raise Exception("maxRunsODE must be set to a finite value >= 0."
                            "(0 means indefinite runs until convergence)")
        # Check that maxTimeODE is in proper format
        if not isinstance(self._sa.maxTimeODE, int) and not isinstance(self._sa.maxTimeODE, float):
            raise TypeError('maxRunsODE needs to be float or int.')
        if self._sa.maxRunsODE == None or self._sa.maxRunsODE < 0 or not np.isfinite(self._sa.maxTimeODE):
            raise Exception("maxRunsODE must be set to a finite value >= 0."
                            "(0 means indefinite run time until until convergence)")
        
        # Check that Q value is positive
        if self._sa.Q < 0:
            raise Exception("Negative Q value (="+str(self._sa.Q)+\
                            "). It needs to be positive.")
    
    def run(self, generator):
        """
        
        """
        
        timeStamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
        filePath = "results/"+str(self._sa.simulationName)+"/"+str(timeStamp)+"/"
        
        # GENERATE
        if self._sa.verbose:
            print("-----------------------------------------------------------")
            print("Generating initial configurations ...")
        generationStart = time()
        gen_rs, gen_vs, gen_TXE, simulations = generator.generate(filePath = filePath,
                                                                  plotInitialConfigs = self._sa.plotInitialConfigs,
                                                                  verbose = self._sa.verbose)
        if self._sa.verbose:
            print("Generation time: %1.1f sec." % (time()-generationStart))
            print("-----------------------------------------------------------")
            if self._sa.useGPU:
                print("Running "+str(simulations)+" simulations on GPU ...")
            else:
                print("Running "+str(simulations)+" simulations on CPU ...")
        
        # SIMULATE
        runStart = time()
        if self._sa.useGPU:
            initGPU(self,
                    simulations = simulations,
                    rs_in = gen_rs,
                    vs_in = gen_vs,
                    TXEs_in = gen_TXE,
                    verbose = self._sa.verbose)
            
            converged = False
            totalRunTimeGPU = 0.0
            gpuruns = 0
            
            Ec0 = [0]*simulations
            for i in range(0,simulations):
                Ec0[i] = self._sa.cint.coulombEnergies(self._sa.Z, gen_rs[i], fissionType_in=self._sa.fissionType)
            
            # Run GPU kernel until all trajectories has converged
            while converged == False:
                gpuruns += 1
                runTimeGPU = runGPU(self, verbose = self._sa.verbose)
                totalRunTimeGPU += runTimeGPU
                
                # Fetch the coordinates from the GPU
                run_rs = getCoordinatesGPU(self)
                
                converged, EcRatio = runHasConverged(self,
                                                     Ec0_in=Ec0,
                                                     rs_in=run_rs,
                                                     simulations_in=simulations)
                if self._sa.verbose:
                    print("GPU run "+str(gpuruns)+": %1.5f\t%1.2f\t%1.1fsec" % (EcRatio, self._sa.minEc, runTimeGPU))

            if self._sa.verbose:
                print("GPU kernel ran "+str(gpuruns)+" times. Total run time: %1.1f sec" % totalRunTimeGPU)

            # Fetch the rest of the results from the GPU
            run_vs = getVelocitiesGPU(self)
            run_status = getStatusGPU(self)
            
            if self._sa.saveKineticEnergies:
                run_ekins = getKineticEnergiesGPU(self)
            else:
                run_ekins = np.zeros(simulations)
            
            if self._sa.saveTrajectories:
                run_trajectories = getTrajectoriesGPU(self)
            else:
                run_trajectories = np.zeros(simulations)
        else:
            run_rs, run_vs, run_status, run_ekins, run_trajectories = runCPU(self,
                                                                             simulations = simulations,
                                                                             rs_in = gen_rs,
                                                                             vs_in = gen_vs,
                                                                             TXEs_in = gen_TXE,
                                                                             verbose = self._sa.verbose)
        if self._sa.verbose:
            print("Simulation time: %1.1f sec." % (time()-runStart))
            print("-----------------------------------------------------------")
            print("Storing data in results/" + str(self._sa.simulationName) + \
                  "/" + str(timeStamp) + "/ ...")
        
        # STORE
        storeStart = time()
        finalErrorCount = storeRunData(self,
                                       rs_in = run_rs,
                                       r0s_in = gen_rs,
                                       vs_in = run_vs,
                                       v0s_in = gen_vs,
                                       TXEs_in = gen_TXE,
                                       status_in = run_status,
                                       ekins_in = run_ekins,
                                       simulations = simulations,
                                       trajectories_in = run_trajectories,
                                       filePath_in = filePath)
        if self._sa.verbose:
            print("Data storing time: %1.1f sec." % (time()-storeStart))
            print("-----------------------------------------------------------")
            print(str(finalErrorCount)+" of "+str(simulations)+" simulations had errors.")
            print("-----------------------------------------------------------")
            print("Files generated:")
            print(str(filePath) + "shelvedVariables.sb")
            if self._sa.saveTrajectories:
                print(str(filePath) + "shelvedTrajectories.sb")
            print("-----------------------------------------------------------")
            print("Total program time: %1.1f sec." % (time()-generationStart))
            # initial configs
    
    def adaptiveRun(self, generator, adaptiveRuns, stepSize):
        """
        
        """
        
        timeStamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
        
        params = [1.0,   # sigma_D
                  2.0,   # sigma_x
                  1.0,   # sigma_y
                  1.0    # sigma_Eff
                 ]
        
        old_devsqrt = 100.0
        print("Starting "+str(adaptiveRuns)+" adaptive runs.")
        for j in range(0, adaptiveRuns):
            filePath = "results/"+str(self._sa.simulationName)+"/"+str(timeStamp)+"/ar_"+str(j+1)+"/"
            
            print(str(j+1)+"/"+str(adaptiveRuns)+": "),
            
            # Randomize generator parameters
            # - mu_d: center or saddle
            # - sigma_D
            # - sigma_x
            # - sigma_y
            # - mu_Eff
            # - sigma_Eff
            params_old = params
            params[j%4] += stepSize*(2.0*np.random.rand() - 1.0)
            generator.setSigmaParams(sigma_D = params[0],
                                     sigma_x = params[1],
                                     sigma_y = params[2],
                                     sigma_EKT_sciss = params[3])
            
            # GENERATE
            generationStart = time()
            gen_rs, gen_vs, gen_TXE, simulations = generator.generate(filePath = filePath,
                                                                      plotInitialConfigs = self._sa.plotInitialConfigs,
                                                                      verbose = False)
            print("["),
            for pp in range(0,len(params)):
                print(" %1.2f" % params[pp]),
            print("]\t"),
            
            # SIMULATE
            runStart = time()
            if self._sa.useGPU:
                run_rs, run_vs, run_status, run_ekins, run_trajectories = runGPU(self,
                                                                                 simulations = simulations,
                                                                                 rs_in = gen_rs,
                                                                                 vs_in = gen_vs,
                                                                                 TXEs_in = gen_TXE,
                                                                                 verbose = False)
            else:
                run_rs, run_vs, run_status, run_ekins = runCPU(self,
                                                               simulations = simulations,
                                                               rs_in = gen_rs,
                                                               vs_in = gen_vs,
                                                               TXEs_in = gen_TXE,
                                                               verbose = False)
            print("%1.1f sec\t" % (time()-runStart)),
            
            # Check goodness of simualtion
            Ec0 = np.zeros([simulations,3])
            Ec = np.zeros([simulations,3])
            Ekin0 = np.zeros([simulations,3])
            Ekin = np.zeros([simulations,3])
            angle = np.zeros(simulations)
            shelveStatus = [0] * simulations
            
            for i in range(0,simulations):
                # Mirror the configuration if the tp goes "through" to the other side
                if self._sa.fissionType != 'BF' and run_rs[i,1] < 0:
                    run_rs[i,1] = -run_rs[i,1]
                    run_rs[i,3] = -run_rs[i,3]
                    run_rs[i,5] = -run_rs[i,5]
                    run_vs[i,1] = -run_vs[i,1]
                    run_vs[i,3] = -run_vs[i,3]
                    run_vs[i,5] = -run_vs[i,5]
                
                Ec0[i] = self._sa.cint.coulombEnergies(self._sa.Z, gen_rs[i], fissionType_in=self._sa.fissionType)
                Ec[i] = self._sa.cint.coulombEnergies(self._sa.Z, run_rs[i], fissionType_in=self._sa.fissionType)
                Ekin0[i] = getKineticEnergies(v_in=gen_vs[i], m_in=self._sa.mff) 
                Ekin[i] = getKineticEnergies(v_in=run_vs[i], m_in=self._sa.mff)
                angle[i] = getAngle(run_vs[i][0:2],run_vs[i][-2:len(run_vs[i])])
                
                # Check that run did not get an error for some reason
                if run_status[i] != 0:
                    shelveStatus[i] = 1
                # Check that Ec is finite
                if not np.isfinite(np.sum(Ec[i])):
                    shelveStatus[i] = 1
                # Check that Ekin is finite
                if not np.isfinite(np.sum(Ekin[i])):
                    shelveStatus[i] = 1
                # Check that Energy is conserved
                if (gen_TXE[i] + np.sum(Ec[i]) + np.sum(Ekin[i])) > self._sa.Q:
                    shelveStatus[i] = 1
                # Check that system has converged
                if np.sum(Ec[i]) > self._sa.minEc*np.sum(Ec0[i]):
                    shelveStatus[i] = 1
            print(str(np.sum(shelveStatus))+"err\t"),
            Etp_inf = np.mean(Ekin[:,0])
            Ehf_inf = np.mean(Ekin[:,1])
            Elf_inf = np.mean(Ekin[:,2])
            Eff_inf = np.mean(Ekin[:,1] + Ekin[:,2])
            Etp_sciss = np.mean(Ekin0[:,0])
            theta = getDistMaxBinCenter(angle,nbins=50)
            
            deviations = [15.8-Etp_inf, 3.0-Etp_sciss, 82.0-theta,
                          63.25-Ehf_inf, 92.85-Elf_inf, 155.5-Eff_inf]
            devsqrt = 0.0
            if self._sa.verbose:
                print("["),
                for d in range(0,len(deviations)):
                    devsqrt = deviations[d]**2
                    print(" %1.2f" % deviations[d]),
                print("]\t%1.3f" % (np.sqrt(devsqrt)))
            if devsqrt < old_devsqrt:
                old_params = params
                old_devsqrt = devsqrt
            else:
                if 0.5*np.exp(old_devsqrt-devsqrt) > np.random.random():
                    old_params = params
                    old_devsqrt = devsqrt
                else:
                    params = old_params
                    devsqrt = old_devsqrt
        # end of adaptiveRuns for-loop

    def checkConfiguration(self, r_in, v_in, TXE_in):
        """
        Check initial configurations for error.
        
        :type r_in: list of floats
        :param r_in: Coordinates of the fission fragments [xtp, ytp, xhf, yhf, xlf, ylf].
        
        :type v_in: list of floats
        :param v_in: Velocities of the fission fragments [vxtp, vytp, vxhf, vyhf, vxlf, vylf].
        
        :type TXE_in: float
        :param TXE_in: Initial excitation energy.
        """
        
        errorCount = 0
        
        Ec = self._sa.cint.coulombEnergies(self._sa.Z, r_in, fissionType_in=self._sa.fissionType)
        Ekin = getKineticEnergies(v_in=v_in, m_in=self._sa.mff)
        
        # Check that Ec is a number

        # Check that minEc is not too high
        if self._sa.minEc >= np.sum(Ec):
            errorCount += 1
            if self._sa.displayGeneratorErrors:
                print("minEc is higher than initial Coulomb energy! ("+\
                      str(self._sa.minEc)+' > '+str(np.sum(Ec))+")")
        
        # Check that Coulomb energy is not too great
        if self._sa.Q < np.sum(Ec):
            errorCount += 1
            if self._sa.displayGeneratorErrors:
                print("Energy not conserved: Particles are too close, "
                      "generating a Coulomb Energy > Q ("+\
                      str(np.sum(Ec))+">"+str(self._sa.Q)+"). Ec="+\
                      str(Ec))
                            
        # Check that total energy is conserved
        if self._sa.Q < (np.sum(Ekin) + np.sum(Ec) + TXE_in):
            errorCount += 1
            if self._sa.displayGeneratorErrors:
                print("Energy not conserved: TXE + Ekin + Ec > Q ("+\
                      str(np.sum(Ec)+TXE_in+np.sum(Ekin))+">"+str(self._sa.Q)+")")
                            
        # Check that r is in proper format
        if self._sa.fissionType == 'BF':
            if len(r_in) != 4:
                errorCount += 1
                if self._sa.displayGeneratorErrors:
                    print("r needs to include 4 initial coordinates, i.e. x "
                          "and y for the fission fragments.")
        else:
            if len(r_in) != 6:
                errorCount += 1
                if self._sa.displayGeneratorErrors:
                    print("r needs to include 6 initial coordinates, i.e. x "
                          "and y for the fission fragments.")
        for i in r_in:
            if not isinstance(i, float) and i != 0:
                errorCount += 1
                if self._sa.displayGeneratorErrors:
                    print("All elements in r must be float, (or atleast int if "
                          "zero).")
            if not np.isfinite(i) or i == None:
                errorCount += 1
                if self._sa.displayGeneratorErrors:
                    print("All elements in r must be set to a finite value.")
            

        # Check that v is in proper format
        if self._sa.fissionType == 'BF':
            if len(v_in) != 4:
                errorCount += 1
                if self._sa.displayGeneratorErrors:
                    print("v needs to include 4 initial velocities, i.e. vx "
                          "and vy for the fission fragments.")
        else:
            if len(v_in) != 6:
                errorCount += 1
                if self._sa.displayGeneratorErrors:
                    print("v needs to include 6 initial velocities, i.e. vx "
                          "and vy for the fission fragments.")
        for i in v_in:
            if not isinstance(i, float) and i != 0:
                errorCount += 1
                if self._sa.displayGeneratorErrors:
                    print("All elements in v must be float, (or atleast int if "
                          "zero).")
            if not np.isfinite(i) or i == None:
                errorCount += 1
                if self._sa.displayGeneratorErrors:
                    print("All elements in v must be set to a finite value.")
        
        
        # Check that particles do not overlap
        if self._sa.fissionType == 'BF':
            if abs(r_in[0] - r_in[2]) <= (self._sa.ab[0] + self._sa.ab[2]):
                errorCount += 1
                if self._sa.displayGeneratorErrors:
                    print("HF and LF are overlapping: ("+\
                          str(abs(r_in[0] - r_in[2]))+" <= "+\
                          str(self._sa.ab[0] + self._sa.ab[2])+"). Increase their "
                          "initial spacing.")      
        else:
            if circleEllipseOverlap(r_in[0:4], self._sa.ab[2], self._sa.ab[3], self._sa.rad[0]):
                errorCount += 1
                if self._sa.displayGeneratorErrors:
                    print("TP and HF are overlapping: ("+\
                          str((r_in[2] - r_in[0])**2/(self._sa.ab[2]+self._sa.rad[0])**2 + \
                              (r_in[3] - r_in[1])**2/(self._sa.ab[3]+self._sa.rad[0])**2)+ \
                          " <= 1). Increase their initial spacing.")
            if circleEllipseOverlap(r_in[0:2]+r_in[4:6], self._sa.ab[4], self._sa.ab[5], self._sa.rad[0]):
                errorCount += 1
                if self._sa.displayGeneratorErrors:
                    print("TP and LF are overlapping: ("+\
                          str((r_in[4] - r_in[0])**2/(self._sa.ab[4]+self._sa.rad[0])**2 + \
                              (r_in[5] - r_in[1])**2/(self._sa.ab[5]+self._sa.rad[0])**2)+ \
                          " <= 1). Increase their initial spacing.")
            if abs(r_in[2] - r_in[4]) <= (self._sa.ab[2] + self._sa.ab[4]):
                errorCount += 1
                if self._sa.displayGeneratorErrors:
                    print("HF and LF are overlapping: ("+str(abs(r_in[2]-r_in[4]))+\
                          " <= "+str(self._sa.ab[2] + self._sa.ab[4])+"). "
                          "Increase their initial spacing.")  

        # Check that total linear momentum is conserved
        # Check that total angular momentum is conserved
        
        return errorCount

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
            
            tdist = [0]*int(len(r[:,0])/2)
            for i in range(1,len(r[0])):
                for j in range(0,int(len(r[:,0])/2)):
                    tdist[j] += np.sqrt((r[j*2,i]-r[j*2,i-1])**2 + (r[j*2+1,i]-r[j*2+1,i-1])**2)
            
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
        maxrad = max(self._sa.ab)
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
            plotEllipse(r[0][i*skipsize],r[1][i*skipsize],self._sa.ab[0],self._sa.ab[1])
            plotEllipse(r[2][i*skipsize],r[3][i*skipsize],self._sa.ab[2],self._sa.ab[3])
            plotEllipse(r[4][i*skipsize],r[5][i*skipsize],self._sa.ab[4],self._sa.ab[5])
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

"""
def _throwException(self, exceptionType_in, exceptionMessage_in):
    Wrapper for throwing exceptions, in order to make it possible to switch
    between interrupting the program or letting it continue on with another
    simulation.
    
    :type exceptionType_in: exception
    :param exceptionType_in: Exception type to throw.
    
    :type exceptionMessage_in: string
    :param exceptionMessage_in: Message to show in exception/simulation status.
    if self._interruptOnException:
        raise exceptionType_in(str(exceptionMessage_in))
    else:
        if self._exceptionCount == 0:
            print(str(exceptionType_in)+': '+str(exceptionMessage_in))
            self._exceptionMessage = exceptionMessage_in
            self._exceptionCount = 1
        else:
            self._exceptionCount += 1
"""

def runCPU(self, simulations, rs_in, vs_in, TXEs_in, verbose):
    """
    Runs simulation by solving the ODE for equations of motions on the CPU
    for the initialized system.
    """
    
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
        
        a_out = self._sa.cint.accelerations(Z_in = self._sa.Z,
                                            r_in = list(u[0:int(len(u)/2)]),
                                            m_in = self._sa.mff,
                                            fissionType_in=self._sa.fissionType)
        
        return list(u[int(len(u)/2):len(u)]) + a_out

    dts = np.arange(0.0, 1000.0, self._dt)
    r_out = np.zeros([simulations,6])
    v_out = np.zeros([simulations,6])
    status_out = [0] * simulations
    if self._sa.saveTrajectories:
        trajectories_out = np.zeros([simulations,6,self._sa.trajectorySaveSize])
    else:
        trajectories_out = np.zeros(simulations)
    
    if self._sa.saveKineticEnergies:
        ekins_out = [0] * simulations
    else:
        ekins_out = []
    
    for i in range(0, simulations):
        runNumber = 0
        errorCount = 0
        errorMessages = []
        ode_matrix = [[],[],[],[],[],[]]
        
        v0 = vs_in[i]
        r0 = rs_in[i]
        r_out[i] = r0
        v_out[i] = v0
        Ec0 = self._sa.cint.coulombEnergies(self._sa.Z, r0, fissionType_in=self._sa.fissionType)
        Ekin0 = getKineticEnergies(v_in=v0, m_in=self._sa.mff)
        Ekin = Ekin0
        Ec = Ec0
        
        if self._sa.fissionType == 'BF':
            D = abs(r0[2])
        else:
            D = abs(r0[4]-r0[2])
        
        if self._sa.saveKineticEnergies:
            ekins_out[i] = [np.sum(Ekin)]
        
        startTime = time()
        while (runNumber < self._sa.maxRunsODE or self._sa.maxRunsODE == 0) and \
              ((time()-startTime) < self._sa.maxTimeODE or self._sa.maxTimeODE == 0) and \
              errorCount == 0 and \
              np.sum(Ec) >= self._sa.minEc*np.sum(Ec0):
            runTime = time()
            runNumber += 1
            
            ode_sol = odeint(odeFunction, (list(r_out[i]) + list(v_out[i])), dts).T
            
            if self._sa.collisionCheck:
                for i in range(0,len(ode_sol[0,:])):
                    if self._sa.fissionType != 'BF' and \
                       circleEllipseOverlap(r_in=[ode_sol[0,i],ode_sol[1,i],
                                                  ode_sol[2,i],ode_sol[3,i]],
                                            a_in=self._sa.ab[2], b_in=self._sa.ab[3],
                                            rad_in=self._sa.rad[0]):
                        errorCount += 1
                        errorMessages.append("TP and HF collided during acceleration!")
                    if self._sa.fissionType != 'BF' and \
                       circleEllipseOverlap(r_in=[ode_sol[0,i],ode_sol[1,i],
                                                  ode_sol[4,i],ode_sol[5,i]],
                                            a_in=self._sa.ab[4], b_in=self._sa.ab[5],
                                            rad_in=self._sa.rad[0]):
                        errorCount += 1
                        errorMessages.append("TP and LF collided during acceleration!")
                    if (ode_sol[-2-int(len(ode_sol[:,0])/2),i]-\
                        ode_sol[-4-int(len(ode_sol[:,0])/2),i]) < \
                                                    (self._sa.ab[-2]+self._sa.ab[-4]):
                        errorCount += 1
                        errorMessages.append("HF and LF collided during acceleration!")
            
            r_out[i] = list(ode_sol[0:int(len(ode_sol)/2),-1])
            v_out[i] = list(ode_sol[int(len(ode_sol)/2):len(ode_sol),-1])
            
            # Get Center of mass coordinates
            x_cm, y_cm = getCentreOfMass(r_in=r_out[i], m_in=self._sa.mff)

            # Update current coulomb energy
            Ec = self._sa.cint.coulombEnergies(Z_in=self._sa.Z, r_in=r_out[i],fissionType_in=self._sa.fissionType)
            
            # Get the current kinetic energy
            Ekin = getKineticEnergies(v_in=v_out[i], m_in=self._sa.mff)
            if self._sa.saveKineticEnergies:
                ekins_out[i].append(np.sum(Ekin))

            # Check that potential and kinetic energies are finite
            if not np.isfinite(np.sum(Ec)):
                errorCount += 1
                errorMessages.append("Coulomb Energy not finite after run "+\
                                     "number "+str(runNumber)+\
                                     "! Ec="+str(np.sum(Ec)))
            if not np.isfinite(np.sum(Ekin)):
                errorCount += 1
                errorMessages.append("Kinetic Energy not finite after"
                                     " run number "+str(runNumber)+\
                                     "! Ekin="+\
                                     str(np.sum(Ekin)))
            
            # Check that energy conservation is not violated
            if (TXEs_in[i] + np.sum(Ekin) + np.sum(Ec)) > self._sa.Q :
                errorCount += 1
                errorMessages.append("Excitation energy + Kinetic + Coulomb "
                                     "energy higher than initial Q-value. This "
                                     "breaks energy conservation! Run: "+\
                                     str(runNumber)+", "+\
                                     str(np.sum(Ekin)+TXEs_in[i]+np.sum(Ec))+">"+\
                                     str(self._sa.Q)+"\tEc: "+str(Ec)+\
                                     " Ekin: "+str(Ekin)+\
                                     " TXE:"+str(TXEs_in[i]))
            
            # Save paths to file to free up memory
            if self._sa.saveTrajectories:
                """if not errorCount > 0:
                    f_data = file(self._filePath + "trajectories_"+\
                                  str(runNumber)+".bin", 'wb')
                    np.save(f_data,np.array([ode_sol[0],ode_sol[1],ode_sol[2],ode_sol[3],ode_sol[4],ode_sol[5],
                                             np.ones(len(ode_sol[0]))*x_cm,
                                             np.ones(len(ode_sol[0]))*y_cm]))
                    f_data.close()
                """
                if len(ode_matrix[0]) < self._sa.trajectorySaveSize:
                    for od in range(0,6):
                        ode_matrix[od].extend(ode_sol[od][0:self._sa.trajectorySaveSize])
            # Free up some memory
            del ode_sol
        # end of while loop
        stopTime = time()
        
        # Throw exception if the maxRunsODE was reached before convergence
        if runNumber == self._sa.maxRunsODE and np.sum(Ec) > self._sa.minEc*np.sum(Ec0):
            errorCount += 1
            errorMessages.append("Maximum allowed runs (maxRunsODE) was reached"
                                 " before convergence.")
            
        if (stopTime-startTime) >= self._sa.maxTimeODE and np.sum(Ec) > self._sa.minEc*np.sum(Ec0):
            errorCount += 1
            errorMessages.append("Maximum allowed runtime (maxTimeODE) was "
                                 "reached before convergence.")
        
        if verbose:
            if errorCount == 0:
                outString = "a: %1.2f\tEi: %1.2f" % (getAngle(v_out[i][0:2],v_out[i][-2:len(v_out[i])]),
                                                     np.sum(Ec0)+np.sum(Ekin0))
            else:
                outString = errorMessages[0]
            print("S: "+str(i+1)+"/"+str(simulations)+"\t"+str(outString))
        status_out[i] = errorCount
        
        if self._sa.saveTrajectories:
            trajectories_out[i] = ode_matrix
    # end of for loop
    print np.shape(trajectories_out)
    return r_out, v_out, status_out, ekins_out, trajectories_out
# end of runCPU()

def initGPU(self, simulations, verbose, rs_in, vs_in, TXEs_in):
    """
    """
    # Import PyOpenCL and related sub packages
    import pyopencl as cl
    import pyopencl.array
    import pyopencl.clrandom
    import pyopencl.clmath
    
    # Preprocessor defines for the compiler of the OpenCL-code
    defines = ""
    if self._sa.GPU64bitFloat:
        defines += "#define ENABLE_DOUBLE\n"
    if self._sa.fissionType == 'BF':
        defines += "#define BINARY_FISSION\n"
    if self._sa.collisionCheck:
        defines += "#define COLLISION_CHECK\n"
    if self._sa.saveTrajectories:
        defines += "#define SAVE_TRAJECTORIES\n"
    
    #if self._sa.betas[0] == 1 and self._sa.betas[1] == 1 and self._sa.betas[2] == 1:
    #    #defines += "#define FULL_SPHERICAL\n"
    
    class DictWithDefault(defaultdict):
        def __missing__(self, key):
            return key + str(" is not defined")
            
    # Constants used in the kernel code will be pasted into the kernel code
    # through the replacements dictionary.
    replacements = DictWithDefault()
    replacements['dt'] = '%e' % self._dt
    replacements['odeSteps'] = '%d' % self._odeSteps
    replacements['trajectorySaveSize'] = '%d' % self._sa.trajectorySaveSize
    replacements['defines'] = defines
    replacements['Q'] = '%e' % self._sa.Q
    replacements['Q12'] = '%e' % (float(self._sa.Z[0]*self._sa.Z[1])*self._ke2)
    replacements['Q13'] = '%e' % (float(self._sa.Z[0]*self._sa.Z[2])*self._ke2)
    replacements['Q23'] = '%e' % (float(self._sa.Z[1]*self._sa.Z[2])*self._ke2)
    #ec, z, m, rad, ab
    replacements['ec2_1'] = '%e' % (self._sa.ec[0]**2)
    replacements['ec2_2'] = '%e' % (self._sa.ec[1]**2)
    replacements['ec2_3'] = '%e' % (self._sa.ec[2]**2)
    replacements['ab1'] = '%e' % self._sa.ab[0]
    replacements['ab2'] = '%e' % self._sa.ab[1]
    replacements['ab3'] = '%e' % self._sa.ab[2]
    replacements['Z1'] = '%d' % self._sa.Z[0]
    replacements['Z2'] = '%d' % self._sa.Z[1]
    replacements['Z3'] = '%d' % self._sa.Z[2]
    replacements['m1i'] = '%e' % (1.0/self._sa.mff[0])
    replacements['m2i'] = '%e' % (1.0/self._sa.mff[1])
    replacements['m3i'] = '%e' % (1.0/self._sa.mff[2])
    replacements['rad1'] = '%e' % self._sa.rad[0]
    replacements['rad2'] = '%e' % self._sa.rad[1]
    replacements['rad3'] = '%e' % self._sa.rad[2]
    
    
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
    self._queue = cl.CommandQueue(self._ctx, properties=queueProperties)
    
    programBuildOptions = "-cl-fast-relaxed-math -cl-mad-enable"
    
    #if verbose:
    #    programBuildOptions += " -cl-nv-verbose"
    if not self._sa.GPU64bitFloat:
        programBuildOptions += " -cl-single-precision-constant"
    
    # Build the kernel program and get the kernel function as an object
    self._prg = (cl.Program(self._ctx, kernelCode)
                     .build(options=programBuildOptions))
    self._kernel = self._prg.gpuODEsolver
    
    # Upload data to GPU
    setGPUdata(self, rs_in=rs_in, vs_in=vs_in, TXEs_in=TXEs_in)
#end of initGPU()

def setGPUdata(self, rs_in, vs_in, TXEs_in):
    """
    """
    # Import PyOpenCL and related sub packages
    import pyopencl as cl
    import pyopencl.array
    import pyopencl.clrandom
    import pyopencl.clmath
    # Allocate memory for coordinates, velocities etc on GPU
    try:
        # Coordinates
        self._r_gpu = cl.array.to_device(self._queue,
               rs_in.astype(np.float64 if self._sa.GPU64bitFloat \
                                       else np.float32))
        # Velocities
        self._v_gpu = cl.array.to_device(self._queue,
               vs_in.astype(np.float64 if self._sa.GPU64bitFloat \
                                       else np.float32))
        # Status of simulation
        self._status_gpu = cl.array.zeros(self._queue,
                                          (self._nbrOfThreads, ),
                                          np.uint32)
        # Kinetic energies throughout the run
        if self._sa.saveKineticEnergies:
            self._ekins_gpu = cl.array.zeros(self._queue,
                                             (self._nbrOfThreads, 1000),
                                             np.float32)
        if self._sa.saveTrajectories:
            self._trajectories_gpu = cl.array.zeros(self._queue,
                                                    (self._nbrOfThreads, int(2*len(self._sa.Z)), self._sa.trajectorySaveSize),
                                                    np.float64 if self._sa.GPU64bitFloat \
                                                    else np.float32)
        # Estimated ODE error size
        #self._errorSize_gpu = cl.array.zeros(self._queue,
        #                                     (self._nbrOfThreads, 12),
        #                                     np.float32)
    except cl.MemoryError:
        raise Exception("Unable to allocate global memory on device,"
                        " out of memory?")
#end of setGPUdata

def runGPU(self, verbose):
    """
    """

    # GPU input data/arguments
    args = [self._r_gpu.data,
            self._v_gpu.data,
            self._status_gpu.data
           ]
    if self._sa.saveKineticEnergies:
        args += [self._errorSize_gpu.data]
    if self._sa.saveTrajectories:
        args += [self._trajectories_gpu.data]
    
    # Run GPU kernel
    self._kernelObj = self._kernel(self._queue,
                                   self._globalSize,
                                   self._localSize,
                                   *args)
    
    # Wait until the threads have finished
    try:
        self._kernelObj.wait()
    except pyopencl.RuntimeError:
        if time() - startTimeGPU > 5.0:
            raise Exception("Kernel runtime error. Over 5 seconds had "
                            "passed when kernel aborted.")
        else:
            raise
    
    # Make sure that the queue is finished
    self._queue.finish()
    
    # Calculate GPU execution time through profiling
    runTimeGPU_out = 1e-9*(self._kernelObj.profile.end - \
                           self._kernelObj.profile.start)
    
    return runTimeGPU_out
# end of runGPU()

def getCoordinatesGPU(self):
    """
    """
    
    return self._r_gpu.get()
# end of getCoordinatesGPU

def getVelocitiesGPU(self):
    """
    """
    
    return self._v_gpu.get()
# end of getVelocitiesGPU

def getStatusGPU(self):
    """
    """
    
    return self._status_gpu.get()
# end of getStatusGPU

def getKineticEnergiesGPU(self):
    """
    """
    
    return self._ekins_gpu.get()
# end of getKineticEnergiesGPU

def getTrajectoriesGPU(self):
    """
    """
    
    return self._trajectories_gpu.get()
# end of getTrajectoriesGPU

def runHasConverged(self, Ec0_in, rs_in, simulations_in):
    """
    """
    
    for i in range(0, simulations_in):
        thisEc = np.sum(self._sa.cint.coulombEnergies(self._sa.Z, rs_in[i], fissionType_in=self._sa.fissionType))
        if thisEc > self._sa.minEc*np.sum(Ec0_in[i]):
            return False, (thisEc/np.sum(Ec0_in[i]))
    
    return True, self._sa.minEc

def storeRunData(self, rs_in, r0s_in, vs_in, v0s_in, TXEs_in, status_in, ekins_in, simulations, trajectories_in, filePath_in=None):
    """
    Store data in a file.
    """
    if filePath_in == None:
        timeStamp = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
        
    # Create results folder if it doesn't exist
    if not os.path.exists(filePath_in):
        os.makedirs(filePath_in)
    
    # Initialize arrays
    Ec0 = [0] * simulations#np.zeros([len(status_out),3])
    Ec = [0] * simulations#np.zeros([len(status_out),3])
    Ekin0 = [0] * simulations#np.zeros([len(status_out),3])
    Ekin = [0] * simulations#np.zeros([len(status_out),3])
    angle = [0] * simulations#np.zeros(len(status_out))
    shelveStatus = [0] * simulations
    shelveError = [[]]*simulations
    wentThrough = [False]*simulations
    Ds = np.zeros(simulations)
    
    for i in range(0,simulations):
        if i%5000 == 0 and i > 0 and self._sa.verbose:
            print(str(i)+" of "+str(simulations)+" simulations prepared for storage.")

        # Mirror the configuration if the tp goes "through" to the other side
        if self._sa.fissionType != 'BF' and rs_in[i,1] < 0:
            rs_in[i,1] = -rs_in[i,1]
            rs_in[i,3] = -rs_in[i,3]
            rs_in[i,5] = -rs_in[i,5]
            vs_in[i,1] = -vs_in[i,1]
            vs_in[i,3] = -vs_in[i,3]
            vs_in[i,5] = -vs_in[i,5]
            wentThrough[i] = True
        else:
            wentThrough[i] = False
        
        Ec0[i] = self._sa.cint.coulombEnergies(self._sa.Z, r0s_in[i], fissionType_in=self._sa.fissionType)
        Ec[i] = self._sa.cint.coulombEnergies(self._sa.Z, rs_in[i], fissionType_in=self._sa.fissionType)
        Ekin0[i] = getKineticEnergies(v_in=v0s_in[i], m_in=self._sa.mff) 
        Ekin[i] = getKineticEnergies(v_in=vs_in[i], m_in=self._sa.mff)
        angle[i] = getAngle(vs_in[i][0:2],vs_in[i][-2:len(vs_in[i])])
        Ds[i] = (r0s_in[i,-2] - r0s_in[i,-4])
        
        # Check that run did not get an error for some reason
        if status_in[i] != 0:
            shelveError[i].append("Trajectory ran into error during "
                                  "simulation.")
            shelveStatus[i] = 1
        # Check that Ec is finite
        if not np.isfinite(np.sum(Ec[i])):
            shelveError[i].append("Ec not finite.")
            shelveStatus[i] = 1
        # Check that Ekin is finite
        if not np.isfinite(np.sum(Ekin[i])):
            shelveError[i].append("Ekin not finite.")
            shelveStatus[i] = 1
        # Check that Energy is conserved
        if (TXEs_in[i] + np.sum(Ec[i]) + np.sum(Ekin[i])) > self._sa.Q:
            shelveError[i].append("Energy not conserved, TXE + TKE > Q: "+\
                               str(TXEs_in[i] + np.sum(Ec[i]) + \
                               np.sum(Ekin[i]))+" > "+str(self._sa.Q)+".")
            shelveStatus[i] = 1
        # Check that system has converged
        if np.sum(Ec[i]) > self._sa.minEc*np.sum(Ec0[i]):
            shelveError[i].append("System has not converged. minEc="+\
                               str(self._sa.minEc)+" > Ec/Ec0="+\
                               str(np.sum(Ec[i])/np.sum(Ec0[i])))
            shelveStatus[i] = 1
        
        
        if shelveStatus[i] == 1:
            shelveError[i] = shelveError[i][0]
            #print shelveError
        
        if shelveStatus[i] == 0:
            shelveError[i] = None
    
    
    if self._sa.fissionType == 'BF':
        particles = []
    else:
        particles = [self._sa.tp]
    particles.extend([self._sa.hf,self._sa.lf])
    
    # Store variables and their final values in a shelved file format
    s = shelve.open(filePath_in + 'shelvedVariables.sb', protocol=pickle.HIGHEST_PROTOCOL)
    try:
        s["0"] = {'simName': self._sa.simulationName,
                  'simulations': simulations,
                  'fissionType': self._sa.fissionType,
                  'Q': self._sa.Q,
                  'D': Ds,
                  'r': rs_in,
                  'v': vs_in,
                  'r0': r0s_in,
                  'v0': v0s_in,
                  'TXE': TXEs_in,
                  'Ec0': Ec0,
                  'Ekin0': Ekin0,
                  'angle': angle,
                  'Ec': Ec,
                  'Ekin': Ekin,
                  'ODEruns': self._odeSteps,
                  'status': shelveStatus,
                  'error': shelveError,
                  #'time': runTimeGPU,
                  'wentThrough': wentThrough,
                  'Ekins': ekins_in,
                  'particles': particles,
                  'coulombInteraction': self._sa.cint,
                  'nuclearInteraction': self._sa.nint,
                  'D0': Ds[0],
                  'ab': self._sa.ab,
                  'ec': self._sa.ec,
                  'GPU': self._sa.useGPU
                }
    finally:
        s.close()
    if self._sa.saveTrajectories:
        s = shelve.open(filePath_in + 'shelvedTrajectories.sb', protocol=pickle.HIGHEST_PROTOCOL)
        try:
            s["0"] = {'simName': self._sa.simulationName,
                      'simulations': simulations,
                      'trajectories': trajectories_in,
                      'odeSteps': self._odeSteps,
                      'nbrOfParticles': int(len(self._sa.Z))
                     }
        finally:
            s.close()
    """
    f_data = file(filePath_in + "dataz.bin", 'wb')
    np.save(f_data,np.array([self._sa.simulationName,i,simulations,self._sa.fissionType,False,self._sa.Q,Ds,rs_in,vs_in,r0s_in,
                  v0s_in,
                  TXEs_in,
                  self._odeSteps,
                  shelveStatus,
                  shelveError,
                  wentThrough,
                  ekins_in,
                  self._sa.simulationName,
                  self._sa.fissionType,
                  particles,
                  self._sa.cint,
                  self._sa.nint,
                  Ds[0],
                  self._sa.ab,
                  self._sa.ec,
                  self._sa.useGPU]))
    f_data.close()
    """
    return np.sum(shelveStatus)
# end of storeRunData

def getKineticEnergies(v_in, m_in):
    """
    Retruns kinetic energies of the particles.
    
    :type v_in: list of floats
    :param v_in: Velocities.
    
    :type m_in: list of floats
    :param m_in. Masses.
    
    :rtype: list of floats
    :returns: A list of the kinetic energies (E=m*v^2/2).
    """
    ekin_out = []
    for i in range(0,len(m_in)):
        ekin_out.append(m_in[i]*(v_in[i*2]**2+v_in[i*2+1]**2)*0.5)
    return ekin_out


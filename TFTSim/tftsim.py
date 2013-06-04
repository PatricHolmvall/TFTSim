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
import sys
import numpy as np
from scipy.integrate import odeint
from time import time
from datetime import datetime
from collections import defaultdict
import os
import copy
import shelve
import pylab as pl
import math

def humanReadableSize(size):
    """
    Converts a size in bytes to human readable form.

    :param number size: Size in bytes to convert

    :rtype: string
    :returns: The size in human readable form.
    """
    if size <= 0 or size >= 10e18:
        return "%.3g" % size + " B"
    else:
        p = int(np.log(size) / np.log(1024))
        names = ["", "ki", "Mi", "Gi", "Ti", "Pi", "Ei"]
        return "%.3g" % (size / 1024.0 ** p) + " " + names[p] + "B"

def u2m(m_in):
    """
    Converts a mass given in (unified) atomic mass units (u) to a mass given in
    (electron Volt)/(speed of light)^2 (eV/c^2)
    through the simple relation 1u = 931.494061(21) MeV/c^2 (Source: wikipedia)
    
    :type m_in: float
    :param m_in: Mass of particle in atomic mass units.

    :rtype: float
    :returns: Mass of particle in MeV/c^2.
    """
    return np.float(m_in) * 931.494061

def crudeNuclearRadius(A_in, r0_in=1.25):
    """
    Calculate the nuclear radius of a particle of atomic mass number A through
    the crude approximation r = r0*A^(1/3), where [r0] = fm.
    
    :type A_in: int
    :parameter A_in: Atomic mass number.
    
    :type r0_in: float
    :parameter r0_in: Radius coefficient that you might want to vary as A varies
                   to get the correct fit.
    
    :rtype: float
    :returns: Nuclear radius in fm.
    """
    return r0_in * (np.float(A_in)**(1.0/3.0))

class SimulateTrajectory:
    """
    Initialization of simulates the trajectories for a given system. Preform
    the actual simulation with SimulateTrajectory.run().
    """

    def __init__(self, sa):
        """
        Pre-process and initialize the simulation data.

        :type sa: :class:`tftsim_args.TFTSimArgs` class
        :param sa: An instance of TFTSimArgs describing what kind of system to
                   simulate.
        """

        self._simulationName = sa.simulationName
        self._pint = copy.copy(sa.pint)
        self._fp = copy.copy(sa.fp)
        self._pp = copy.copy(sa.pp)
        self._tp = copy.copy(sa.tp)
        self._hf = copy.copy(sa.hf)
        self._lf = copy.copy(sa.lf)
        self._r = sa.r
        self._minEc = sa.minEc
        self._lostNeutrons = sa.lostNeutrons
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
        
        # Check that simulationName is a string
        if not isinstance(self._simulationName, basestring):
            _throwException('TypeError', 'simulationName must be a string.')
        
        # Check that lost neutron number is in proper format
        if not isinstance(self._lostNeutrons, int):
            _throwException('TypeError', 'lostNeutrons must be an integer.')
        if self._lostNeutrons == None or self._lostNeutrons < 0:
            _throwException('Exception','lostNeutrons must be set to a value >= 0.')

        # Check that particle number is not bogus
        if (self._lostNeutrons + self._tp.A + self._hf.A + self._lf.A) > (self._fp.A + self._pp.A):
            _throwException('Exception',"More particles coming out of fission than in! "+\
                            str(self._fp.A)+"+"+str(self._pp.A)+" < "+\
                            str(self._lostNeutrons)+"+"+str(self._tp.A)+"+"+\
                            str(self._hf.A)+"+"+str(self._lf.A))
        if (self._lostNeutrons + self._tp.A + self._hf.A + self._lf.A) < (self._fp.A + self._pp.A):
            _throwException('Exception',"More particles coming into fission than out! "+\
                            str(self._fp.A)+"+"+str(self._pp.A)+" > "+\
                            str(self._lostNeutrons)+"+"+str(self._tp.A)+"+"+\
                            str(self._hf.A)+"+"+str(self._lf.A))

        # Check that particles are correctly ordered after increasing size
        if self._tp.A > self._hf.A:
            _throwException('Exception',"Ternary particle is heavier than the heavy fission"
                            " fragment!")
        if self._tp.A > self._lf.A:
            _throwException('Exception',"Ternary particle is heavier than the light fission"
                            " fragment!")
        if self._lf.A > self._hf.A:
            _throwException('Exception',"Light fission fragment is heavier than the heavy "
                            "fission fragment!")

        # Check that minEc is in proper format
        if not isinstance(self._minEc, float) and not isinstance(self._minEc, int):
            _throwException('TypeError','minEc needs to be float or int.')
        if self._minEc == None or self._minEc <= 0:
            _throwException('Exception','minEc must be set to a value > 0.')

        self._Ec = self._pint.coulombEnergies(self._Z, self._r)

        self._v = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._Ekin = getKineticEnergies(self)

        # Check that Ec is a number

        # Check that minEc is not too high
        if self._minEc >= np.sum(self._Ec):
            _throwException('Exception',"minEc is higher than initial Coulomb"
                            " energy! ("+str(self._minEc)+' > '+str(np.sum(self._Ec))+")")

        self._Q = getQValue(self)
                                  
        # Check that Q value is reasonable
        if self._Q < 0:
            _throwException('Exception','Negative Q value (='+str(self._Q)+"). It needs to"
                            " be positive.")
        
        # Check that Coulomb energy is not too great
        if self._Q < np.sum(self._Ec):
            _throwException('Exception',"Energy not conserved: Particles are too close, "
                            "generating a Coulomb Energy > Q.")
                            
        # Check that r is in proper format
        if len(self._r) != 6:
            _throwException('Exception',"r needs to include 6 initial coordinates, i.e."
                            " x and y for the fission fragments.")
        for i in self._r:
            if not isinstance(i, float) and i != 0:
                _throwException('TypeError','All elements in r must be float, (or int if zero).')
        
        # Check that particles do not overlap
        if _getDistance(self._r[0:4]) < (self._rtp + self._rhf):
            _throwException('Exception',"TP and HF are overlapping: r=<r_tp+r_hf"
                            " ("+str(_getDistance(self._r[0:4]))+"<"+str(self._rtp+self._rhf)+"). "
                            "Increase their initial spacing.")
        if _getDistance(self._r[0:2]+self._r[4:6]) < (self._rtp + self._rlf):
            _throwException('Exception',"TP and LF are overlapping: r=<r_tp+r_lf"
                            " ("+str(_getDistance(self._r[0:2]+self._r[4:6]))+"<"+str(self._rtp+self._rlf)+"). "
                            "Increase their initial spacing.")
        if _getDistance(self._r[2:6]) < (self._rhf + self._rlf):
            _throwException('Exception',"HF and LF are overlapping: r=<r_hf+r_lf"
                            " ("+str(_getDistance(self._r[2:6]))+"<"+str(self._rhf+self._rlf)+"). "
                            "Increase their initial spacing.")
        
        # Assign initial speeds with remaining kinetic energy
        
        # Check that total angular momentum is conserved
    
    def run(self):
        """
        Runs simulation by solving the ODE for the initialized system.
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


        if self._verbose:
            print("------Starting simulation with initial configuration------")
            print("Reaction: "+str(self._fp.name)+"+"+str(self._pp.name)+" -> "
                  ""+str(self._hf.name)+"+"+str(self._lf.name)+"+"+\
                  str(self._tp.name))
            print("Q-Value: "+str(self._Q)+" MeV/c^2")
            print("Ec,0: "+str(np.sum(self._Ec))+" MeV/c^2\tStop at Ec="+str(self._minEc)+" MeV/c^2")
            print("Ekin,0: "+str(np.sum(self._Ekin))+" MeV/c^2")
            print("r(tp,hf,lf): "+str(self._r))
            
        runNumber = 0
        startTime = time()
        dt = np.arange(0.0, 1000.0, 0.01)
        timeStamp  = datetime.now().strftime("%Y-%m-%d/%H.%M.%S")
        filePath = "results/" + self._simulationName + "/" + timeStamp + "/"
        self._filePath = filePath
        
        # Create results folder if it doesn't exist
        if not os.path.exists(filePath):
            os.makedirs(filePath)
        
        # Create human readable file with system information
        with open(filePath + "systemInfo.txt", 'ab') as f_info:
            f_info.write("---------- System Info ----------\n")
            f_info.write("Reaction: "+str(self._fp.name)+"("+ \
                         str(self._pp.name)+",f) -> "+str(self._hf.name)+"+"+ \
                         str(self._lf.name)+"+"+str(self._tp.name)+"\n")
            f_info.write("Q-value: "+str(self._Q)+"\n")
            f_info.write("Ec,0: "+str(self._Ec)+"\n")
            f_info.write("Ekin,0: "+str(self._Ekin)+"\n")
            f_info.write("r0: "+str(self._r)+"\n")
            f_info.write("v0: "+str(self._v)+"\n")

        # Store variables and their initial values in a shelved file format
        s = shelve.open(filePath + 'shelvedVariables.sb')
        try:
            s['initial'] = { 'angle': _getAngle(self._r[0:2] + self._r[4:6]), 'Ec': self._Ec, 'Ekin': self._Ekin, 'Q': self._Q, 'r': self._r }
        finally:
            s.close()

        for i in range(1,3):
            runTime = time()
            runNumber += 1
            xtp, ytp, xhf, yhf, xlf, ylf, vxtp, vytp, vxhf, vyhf, vxlf, vylf = \
                odeint(odeFunction, (self._r + self._v), dt).T

            self._r = [xtp[-1],ytp[-1],xhf[-1],yhf[-1],xlf[-1],ylf[-1]]
            self._v = [vxtp[-1],vytp[-1],vxhf[-1],vyhf[-1],vxlf[-1],vylf[-1]]

            del vxtp, vytp, vxhf, vyhf, vxlf, vylf
            
            # Check if Coulomb energy is below the set threshold
            self._Ec = self._pint.coulombEnergies(self._Z, self._r)
            
            # Get the current kinetic energy
            self._Ekin = getKineticEnergies(self)

            # Check that none of the particles behave oddly
            if not np.isfinite(np.sum(self._Ec)):
                _throwException('Exception',"Coulomb Energy not finite after run number "+str(runNumber)+"! Ec="+str(np.sum(self._Ec)))
            if not np.isfinite(np.sum(self._Ekin)):
                _throwException('Exception',"Kinetic Energy not finite after run number "+str(runNumber)+"! Ekin="+str(np.sum(self._Ekin)))
            
            # Check that kinetic energy is reasonable compared to Q-value
            if np.sum(self._Ekin) > self._Q:
                _throwException('Exception',"Kinetic Energy is higher than initial Q-value."
                                " This breaks energy conservation!")
            if (np.sum(self._Ekin) + np.sum(self._Ec)) > self._Q :
                _throwException('Exception',"Kinetic + Coulomb energy higher than initial "
                                "Q-value. This breaks energy conservation!")
            
            # Print info about the run to the terminal, if verbose mode is on
            if self._verbose:
                print('Run:'+str(runNumber)+'\ttsteps:'+str(len(xtp))+'\tEc: '+str(np.sum(self._Ec))+'\tEkin: '+str(np.sum(self._Ekin))+
                      '\tTime: '+str(time()-runTime)+'\tAngle: '+str(_getAngle(self._r[0:2]+self._r[4:6])))
            # Halt the simulation if it has been running for too long without converging
            
            # Save paths to file to free up memory
            if self._saveTrajectories:
                if not self._exceptionCount > 0:
                    f_data = file(filePath + "trajectories_"+str(runNumber)+".bin", 'wb')
                    np.save(f_data,np.array([xtp,ytp,xhf,yhf,xlf,ylf]))
                    f_data.close()
                """
                with open(filePath + "trajectory.tsv", 'a') as f_data:
                    # Add a heading describing the columns if file is empty.
                    if  os.path.getsize(filePath + "trajectory.tsv") == 0:
                        f_data.write("#x_tp\ty_tp\tx_hf\ty_hf\tx_lf\ty_hf\n")
                    
                    csvFile = csv.writer(f_data, delimiter='\t', lineterminator='\n')
                    for i in range(0,len(xtp)):
                        csvFile.writerow([xtp[i],ytp[i],xhf[i],yhf[i],xlf[i],ylf[i]])
                """
                """
                with open(filePath + 'trajectory.tsv','r') as csvinput:
                    with open(filePath + 'trajectory.tsv', 'w') as csvoutput:
                        writer = csv.writer(csvoutput, delimiter='\t', lineterminator='\n')
                        reader = csv.reader(csvinput)

                        all = []
                        row = next(reader)
                        row.append('Berry')
                        all.append(row)

                        for row in reader:
                            row.append(row[0])
                            all.append(row)

                        writer.writerows(all)
                """
            # Free up some
            del xtp, ytp, xhf, yhf, xlf, ylf
        
            if self._exceptionCount == 0:
                # Store relevant variables for this run in a shelved file format
                s = shelve.open(filePath + 'shelvedVariables.sb')
                try:
                    s['run'+str(runNumber)] = { 'angle': _getAngle(self._r[0:2] + self._r[4:6]), 'Ec': self._Ec, 'Ekin': self._Ekin }
                finally:
                    s.close()
        # end of loop
        
        
        # Store variables and their final values in a shelved file format
        s = shelve.open(filePath + 'shelvedVariables.sb')
        if self._exceptionCount == 0:
            try:
                s['final'] = { 'angle': _getAngle(self._r[0:2] + self._r[4:6]), 'Ec': self._Ec, 'Ekin': self._Ekin, 'Status': 1, 'runs': runNumber}
            finally:
                s.close()
        else:
            try:
                s['final'] = { 'Status': 0, 'Error': self._exceptionMessage}
            finally:
                s.close()
        
        # Store final info in a human readable text file
        with open(filePath + "systemInfo.txt", 'ab') as f_data:
            if self._exceptionCount == 0:
                # Add final energies if no error occured
                f_data.write("Ec,final: "+str(self._Ec)+"\n")
                f_data.write("Ekin,final: "+str(self._Ekin)+"\n")
                f_data.write("Angle: " +str(_getAngle(self._r[0:2] + \
                                                      self._r[4:6]))+"\n")
                f_data.write("Total run time: "+str(time()-startTime)+"\n")
                f_data.write("Amount of runs: "+str(runNumber)+"\n")
            else:
                # Add exceptionMessage to systemInfo.txt if there was an error
                f_data.write("Error: "+str(self._exceptionMessage)+"\n")
    # end of run()
    def plotTrajectories(self):
        """
        Plot trajectories.
        """
    
    def animateTrajectories(self):
        """
        Animate trajectories.
        """

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
    if self._exceptionInterrupt:
        raise Exception(str(exceptionMessage_in))
    else:
        if self._exceptionCount == 0:
            self._exceptionMessage = exceptionMessage_in
            self._exceptionCount = 1
        else:
            self._exceptionCount += 1


def getQValue(self):
    """
    Calculate the Q value of a given fission process.
    
    :rtype: float
    :returns: Q-value for a given fission process, in MeV/c^2
    """
    
    mEx_neutron = 8.071 # Excess mass of the neutron in MeV/c^2
    
    return np.float(self._fp.mEx + self._pp.mEx - self._tp.mEx -
                    self._hf.mEx - self._lf.mEx - self._lostNeutrons*mEx_neutron) 

def getKineticEnergies(self):
    """
    Retruns kinetic energies of the particles.

    :rtype: A list of the kinetic energies (E=m*v^2/2):
            [E_tp, E_hf, E_lf]
    :returns: list of floats
    """
    return [self._mff[0]*(self._v[0]**2+self._v[1]**2)*0.5,
            self._mff[1]*(self._v[2]**2+self._v[3]**2)*0.5,
            self._mff[2]*(self._v[4]**2+self._v[5]**2)*0.5]

def _generateInitialVelocities(self):
    """
    Generate initial velocities for fission fragments under conservation
    of Energy and Total Angular Momentum. This means that configurations
    where the Coulomb energy is too high is revoked (when particles
    start too close).
    
    :rtype: list of floats
    :returns: List of initial velocities
              [vx_tp, vy_tp, vx_hf, vy_hf, vx_lf, vy_hf]
    """
    # Remember to conserve Energy and Total Angular Momentum
    return 0

def _getDistance(r_in):
    """
    Get distance between two vectors.
    
    :type r_in: list of floats
    :param r_in: List containing coordinates [x1, y1, x2, y2]
    
    :rtype: float
    :returns: Distance between two vectors.
    """
    
    return np.sqrt((r_in[0]-r_in[2])**2+(r_in[1]-r_in[3])**2)

def _getAngle(r_in):
    """
    Get angle between two vectors.
    
    :type r_in: list of floats
    :param r_in: List containing coordinates [x1, y1, x2, y2]
    
    :rtype: float
    :returns: Angle between two vectors.
    """
    
    x1, y1 = r_in[0:2]
    x2, y2 = r_in[2:4]
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.acos(inner_product/(len1*len2))*180.0/np.pi


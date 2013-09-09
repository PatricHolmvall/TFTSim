from __future__ import division
import sys
import numpy as np
from scipy.integrate import odeint
from time import time
from time import sleep
from datetime import datetime
from collections import defaultdict
from scipy.constants import codata
import commands
import os
import copy
import pickle
import shelve
import pylab as pl
import matplotlib.pyplot as plt
import math

_y_r0 = 1.25 # fm
_y_a = 0.68 # fm
_y_as = 21.13 # MeV
_y_w = 2.3

A = [48, 132, 70]
Z = [20, 50, 28]
N = [28, 82, 42]
#A = [4, 134, 96]
#Z = [2, 52, 38]
#N = [2, 82, 58]
rad = [_y_r0*(A[0])**(1.0/3.0),
       _y_r0*(A[1])**(1.0/3.0),
       _y_r0*(A[2])**(1.0/3.0)]
mff = [np.float(A[0]) * codata.value('atomic mass constant energy equivalent in MeV'),
       np.float(A[1]) * codata.value('atomic mass constant energy equivalent in MeV'),
       np.float(A[2]) * codata.value('atomic mass constant energy equivalent in MeV')]

_y_I1 = float(N[0]-Z[0])/float(A[0])
_y_I2 = float(N[1]-Z[1])/float(A[1])
_y_I3 = float(N[2]-Z[2])/float(A[2])
_y_zeta1 = rad[0] / _y_a
_y_zeta2 = rad[1] / _y_a
_y_zeta3 = rad[2] / _y_a
_y_g1 = _y_zeta1*np.cosh(_y_zeta1)-np.sinh(_y_zeta1)
_y_g2 = _y_zeta2*np.cosh(_y_zeta2)-np.sinh(_y_zeta2)
_y_g3 = _y_zeta3*np.cosh(_y_zeta3)-np.sinh(_y_zeta3)
_y_f1 = _y_zeta1**2 * np.sinh(_y_zeta1)
_y_f2 = _y_zeta2**2 * np.sinh(_y_zeta2)
_y_f3 = _y_zeta3**2 * np.sinh(_y_zeta3)
_y_a1 = _y_as*(1.0 - _y_w*(_y_I1**2))
_y_a2 = _y_as*(1.0 - _y_w*(_y_I2**2))
_y_a3 = _y_as*(1.0 - _y_w*(_y_I3**2))
    
YA_12 = (-4.0 * (_y_a/_y_r0)**2 * np.sqrt(_y_a1*_y_a2))
YA_13 = (-4.0 * (_y_a/_y_r0)**2 * np.sqrt(_y_a1*_y_a3))
YA_23 = (-4.0 * (_y_a/_y_r0)**2 * np.sqrt(_y_a2*_y_a3))
YB_12 = (_y_g1*_y_g2)
YB_13 = (_y_g1*_y_g3)
YB_23 = (_y_g2*_y_g3)
YC_12 = (-(_y_g1*_y_f2 + _y_g2*_y_f1))
YC_13 = (-(_y_g1*_y_f3 + _y_g3*_y_f1))
YC_23 = (-(_y_g2*_y_f3 + _y_g3*_y_f2))


def YukawaPlusForce(x12, x13, x23):
    eta_12 = x12 / _y_a
    eta_13 = x13 / _y_a
    eta_23 = x23 / _y_a
    
    F_12 = YA_12 * (YB_12*(eta_12 + 2.0)**2 + YC_12*(eta_12 + 1.0)) * np.exp(-eta_12) / (_y_a * (eta_12)**2)
    F_13 = YA_13 * (YB_13*(eta_13 + 2.0)**2 + YC_13*(eta_13 + 1.0)) * np.exp(-eta_13) / (_y_a * (eta_13)**2)
    F_23 = YA_23 * (YB_23*(eta_23 + 2.0)**2 + YC_23*(eta_23 + 1.0)) * np.exp(-eta_23) / (_y_a * (eta_23)**2)
    return F_12, F_13, F_23

def YukawaPlusPotential(x12, x13, x23):
    eta_12 = x12 / _y_a
    eta_13 = x13 / _y_a
    eta_23 = x23 / _y_a
    P_12 = YA_12 * (YB_12*(4.0 + eta_12) + YC_12) * np.exp(-eta_12) / eta_12
    P_13 = YA_13 * (YB_13*(4.0 + eta_13) + YC_13) * np.exp(-eta_13) / eta_13
    P_23 = YA_23 * (YB_23*(4.0 + eta_23) + YC_23) * np.exp(-eta_23) / eta_23
    return P_12, P_13, P_23

ke2 = 1.43996158

def CoulombForce(x12, x13, x23):
    F_12 = ke2 * Z[0] * Z[1] / x12**2
    F_13 = ke2 * Z[0] * Z[2] / x13**2
    F_23 = ke2 * Z[1] * Z[2] / x23**2
    return F_12, F_13, F_23

def CoulombPotential(x12, x13, x23):
    E_12 = ke2 * Z[0] * Z[1] / x12
    E_13 = ke2 * Z[0] * Z[2] / x13
    E_23 = ke2 * Z[1] * Z[2] / x23
    return E_12, E_13, E_23

"""
num = 10000
xs = np.linspace(0.0, 100.0, num)
fs1 = np.zeros(num)
fs2 = np.zeros(num)
fs3 = np.zeros(num)
f1 = np.zeros(num)
f2 = np.zeros(num)
f3 = np.zeros(num)
ps1 = np.zeros(num)
ps2 = np.zeros(num)
ps3 = np.zeros(num)
p1 = np.zeros(num)
p2 = np.zeros(num)
p3 = np.zeros(num)

for i in range(0,num):
    fs1[i], fs2[i], fs3[i] = YukawaPlusForce(xs[i] + rad[0] + rad[1],
                                             xs[i] + rad[0] + rad[2],
                                             xs[i] + rad[1] + rad[2])
    ps1[i], ps2[i], ps3[i] = YukawaPlusPotential(xs[i] + rad[0] + rad[1],
                                                 xs[i] + rad[0] + rad[2],
                                                 xs[i] + rad[1] + rad[2])
    f1[i], f2[i], f3[i] = CoulombForce(xs[i] + rad[0] + rad[1],
                                       xs[i] + rad[0] + rad[2],
                                       xs[i] + rad[1] + rad[2])
    p1[i], p2[i], p3[i] = CoulombPotential(xs[i] + rad[0] + rad[1],
                                           xs[i] + rad[0] + rad[2],
                                           xs[i] + rad[1] + rad[2])

plt.figure(1)
plt.plot(xs,fs1,'r--',lw=3.0,label=r'Yukawa: $^{48}$Ca + $^{132}$Sn')
plt.plot(xs,fs2,'g--',lw=3.0,label=r'Yukawa: $^{48}$Ca + $^{70}$Ni')
plt.plot(xs,fs3,'b--',lw=3.0,label=r'Yukawa: $^{132}$Sn + $^{70}$Ni')
plt.plot(xs,f1,'r:',lw=3.0,label=r'Coulomb: $^{48}$Ca + $^{132}$Sn')
plt.plot(xs,f2,'g:',lw=3.0,label=r'Coulomb: $^{48}$Ca + $^{70}$Ni')
plt.plot(xs,f3,'b:',lw=3.0,label=r'Coulomb: $^{132}$Sn + $^{70}$Ni')
plt.plot(xs,(fs1+f1),'r-',lw=3.0,label=r'Both: $^{48}$Ca + $^{132}$Sn')
plt.plot(xs,(fs2+f2),'g-',lw=3.0,label=r'Both: $^{48}$Ca + $^{70}$Ni')
plt.plot(xs,(fs3+f3),'b-',lw=3.0,label=r'Both: $^{132}$Sn + $^{70}$Ni')
plt.title('Yukawa Plus Force')
plt.xlabel('Tip Distance [fm]')
plt.ylabel('Energy [MeV/fm]')
plt.legend(loc=4)

plt.figure(2)
plt.plot(xs,ps1,'r--',lw=3.0,label=r'Yukawa: $^{48}$Ca + $^{132}$Sn')
plt.plot(xs,ps2,'g--',lw=3.0,label=r'Yukawa: $^{48}$Ca + $^{70}$Ni')
plt.plot(xs,ps3,'b--',lw=3.0,label=r'Yukawa: $^{132}$Sn + $^{70}$Ni')
plt.plot(xs,p1,'r:',lw=3.0,label=r'Yukawa: $^{48}$Ca + $^{132}$Sn')
plt.plot(xs,p2,'g:',lw=3.0,label=r'Yukawa: $^{48}$Ca + $^{70}$Ni')
plt.plot(xs,p3,'b:',lw=3.0,label=r'Yukawa: $^{132}$Sn + $^{70}$Ni')
plt.plot(xs,(ps1+p1),'r-',lw=3.0,label=r'Both: $^{48}$Ca + $^{132}$Sn')
plt.plot(xs,(ps2+p2),'g-',lw=3.0,label=r'Both: $^{48}$Ca + $^{70}$Ni')
plt.plot(xs,(ps3+p3),'b-',lw=3.0,label=r'Both: $^{132}$Sn + $^{70}$Ni')
plt.title('Yukawa Plus Potential Energy')
plt.xlabel('Tip Distance [fm]')
plt.ylabel('Potential [MeV]')
plt.legend(loc=4)

plt.show()
"""

simulations = 1
trajectorySaveSize = 200000
collisionCheck = True
dt = 0.01
minEc = 0.02

x = 15.0
D = 30.0
y = 5.0

r0 = [0,5,-10,0,30,0]
v0 = [0, 0, 0, 0, 0, 0]

def potentialEnergies(r_in_c):
    d12_c = np.sqrt((r_in_c[0]-r_in_c[2])**2 + (r_in_c[1]-r_in_c[3])**2)
    d13_c = np.sqrt((r_in_c[0]-r_in_c[4])**2 + (r_in_c[1]-r_in_c[5])**2)
    d23_c = np.sqrt((r_in_c[2]-r_in_c[4])**2 + (r_in_c[3]-r_in_c[5])**2)
    Y12, Y13, Y23 = YukawaPlusPotential(x12=d12_c,
                                        x13=d13_c,
                                        x23=d23_c)
    C12, C13, C23 = CoulombPotential(x12=d12_c,
                                     x13=d13_c,
                                     x23=d23_c)
    
    return [[Y12, Y13, Y23], [C12, C13, C23]]

def kineticEnergies(v_in, m_in):
    return 0.5*(m_in[0]*(v_in[0]**2+v_in[1]**2) + \
                m_in[1]*(v_in[2]**2+v_in[3]**2) + \
                m_in[2]*(v_in[4]**2+v_in[5]**2))

def accelerations(Z_in, r_in, m_in):
    # Projected distances
    r12x = r_in[0] - r_in[2]
    r12y = r_in[1] - r_in[3]
    r13x = r_in[0] - r_in[4]
    r13y = r_in[1] - r_in[5]
    r23x = r_in[2] - r_in[4]
    r23y = r_in[3] - r_in[5]
    
    # Absolute distances
    d12 = np.sqrt(r12x**2 + r12y**2)
    d13 = np.sqrt(r13x**2 + r13y**2)
    d23 = np.sqrt(r23x**2 + r23y**2)
    
    # Coulomb forces
    C12x = ke2 * Z[0] * Z[1] * r12x / d12**3
    C12y = ke2 * Z[0] * Z[1] * r12y / d12**3
    C13x = ke2 * Z[0] * Z[2] * r13x / d13**3
    C13y = ke2 * Z[0] * Z[2] * r13y / d13**3
    C23x = ke2 * Z[1] * Z[2] * r23x / d23**3
    C23y = ke2 * Z[1] * Z[2] * r23y / d23**3
    
    # Yukawa forces
    Y12r, Y13r, Y23r = YukawaPlusForce(d12, d13, d23)
    Y12x = r12x * Y12r / d12
    Y12y = r12y * Y12r / d12
    Y13x = r13x * Y13r / d13
    Y13y = r13y * Y13r / d13
    Y23x = r23x * Y23r / d23
    Y23y = r23y * Y23r / d23
    
    # Total forces
    F12x = Y12x + C12x
    F12y = Y12y + C12y
    F13x = Y13x + C13x
    F13y = Y13y + C13y
    F23x = Y23x + C23x
    F23y = Y23y + C23y
    
    # Accelerations
    a1x = ( F12x + F13x)/m_in[0]
    a1y = ( F12y + F13y)/m_in[0]
    a2x = (-F12x + F23x)/m_in[1]
    a2y = (-F12y + F23y)/m_in[1]
    a3x = (-F13x - F23x)/m_in[2]
    a3y = (-F13y - F23y)/m_in[2]
    return [a1x,a1y,a2x,a2y,a3x,a3y]


def plotEllipse(x0_in,y0_in,a_in,b_in):#,color_in,lineStyle_in,lineWidth_in):
    phi = np.linspace(0.0,2*np.pi,100)
    na=np.newaxis
    x_line = x0_in + a_in*np.cos(phi[:,na])
    y_line = y0_in + b_in*np.sin(phi[:,na])
    plt.plot(x_line,y_line,'b-', linewidth=3.0)

def animateTrajectories(rs_ani):
    
    r = rs_ani
    plt.ion()
    maxrad = max(rad)
    plt.axis([np.floor(np.amin([r[0],r[2],r[4]]))-maxrad,
              np.ceil(np.amax([r[0],r[2],r[4]]))+maxrad,
              min(np.floor(np.amin([r[1],r[3],r[5]])),-maxrad)-maxrad,
              max(np.amax([r[1],r[3],r[5]]),maxrad)+maxrad])
    
    skipsize = 2500
    for i in range(0,int(len(r[0])/skipsize)):
        plt.clf()
        plt.axis([np.floor(np.amin([r[0],r[2],r[4]]))-maxrad,
                  np.ceil(np.amax([r[0],r[2],r[4]]))+maxrad,
                  min(np.floor(np.amin([r[1],r[3],r[5]])),-maxrad)-maxrad,
                  max(np.amax([r[1],r[3],r[5]]),maxrad)+maxrad])
        plotEllipse(r[0][i*skipsize],r[1][i*skipsize],rad[0],rad[0])
        plotEllipse(r[2][i*skipsize],r[3][i*skipsize],rad[1],rad[1])
        plotEllipse(r[4][i*skipsize],r[5][i*skipsize],rad[2],rad[2])
        plt.plot(r[0][0:i*skipsize],r[1][0:i*skipsize],'r-',lw=2.0)
        plt.plot(r[2][0:i*skipsize],r[3][0:i*skipsize],'g:',lw=4.0)
        plt.plot(r[4][0:i*skipsize],r[5][0:i*skipsize],'b--',lw=2.0)
        
        plt.draw()
    plt.show()


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
    
    a_out = accelerations(Z_in = Z,
                          r_in = list(u[0:int(len(u)/2)]),
                          m_in = mff)
    
    return list(u[int(len(u)/2):len(u)]) + a_out

dts = np.arange(0.0, 1000.0, dt)
r_out = np.zeros([simulations,6])
v_out = np.zeros([simulations,6])
status_out = [0] * simulations
trajectories_out = np.zeros([simulations,6,trajectorySaveSize])
trajectories_out = np.zeros([simulations,6,trajectorySaveSize])

for i in range(0, simulations):
    runNumber = 0
    errorCount = 0
    errorMessages = []
    ode_matrix = [[],[],[],[],[],[]]
    
    r_out[i] = r0
    v_out[i] = v0
    Ec0 = potentialEnergies(r_in_c = r0)
    Ekin0 = kineticEnergies(v_in = v0, m_in = mff)
    Ekin = Ekin0
    Ec = Ec0
    
    print(r0)
    print(v0)
    print(Ekin0)
    print(Ec0)
    print(np.sum(Ec0))
    print('-------------------------------------------------------------------')
    
    
    D = abs(r0[4]-r0[2])
    
    startTime = time()
    while (np.sum(Ec) >= minEc*np.sum(Ec0)) and errorCount == 0 and runNumber < 1000:
        runTime = time()
        runNumber += 1
        
        ode_sol = odeint(odeFunction, (list(r_out[i]) + list(v_out[i])), dts).T
        
        r_out[i] = list(ode_sol[0:int(len(ode_sol)/2),-1])
        v_out[i] = list(ode_sol[int(len(ode_sol)/2):len(ode_sol),-1])
        
        # Update current coulomb energy
        Ec = potentialEnergies(r_in_c=r_out[i])
        
        # Get the current kinetic energy
        Ekin = kineticEnergies(v_in=v_out[i], m_in=mff)
 
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
        
        # Save paths to file to free up memory
        if len(ode_matrix[0]) < trajectorySaveSize:
            for od in range(0,6):
                ode_matrix[od].extend(ode_sol[od][0:trajectorySaveSize:1])
        # Free up some memory
        del ode_sol
        if runNumber % 1000 == 0:
            print('Run: '+str(runNumber)+',\t\t'+str(np.sum(Ec)/np.sum(Ec0))+'\t\t'+str((time()-startTime)/float(runNumber)))
            print(Ec)
            print(r_out)
            plt.plot(ode_matrix[0],ode_matrix[1],'r-')
            plt.plot(ode_matrix[2],ode_matrix[3],'g-')
            plt.plot(ode_matrix[4],ode_matrix[5],'b-')
            plt.show()
    # end of while loop
    stopTime = time()
    if errorCount > 0:
        print('Error!')
        print(errorMessages)
    else:
        #plt.plot(ode_matrix[0],ode_matrix[1],'r-')
        #plt.plot(ode_matrix[2],ode_matrix[3],'g-')
        #plt.plot(ode_matrix[4],ode_matrix[5],'b-')
        #plt.show()
        animateTrajectories(rs_ani=ode_matrix)
        print('Time: %1.2f s' % (stopTime-startTime))



"""
def plotTrajectories():

    with open(filePath + "trajectories_1.bin","rb") as f_data:
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
        
"""

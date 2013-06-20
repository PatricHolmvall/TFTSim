
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
    
    
    return [u[6], u[7], u[8], u[9], u[10], u[11],
            0.0,0.0,0.0,0.0,0.0,0.0]


dt = np.arange(0.0, 1000.0, 0.01)
r = [0,0,-10,-10,10,10]
Ekin = 10 # MeV
m = [4 * 931.494061, 134 * 931.494061, 96 * 931.494061] # MeV/c^2

v = [np.sqrt(Ekin*2/m[0]),0,np.sqrt(Ekin*2/m[1]),0,np.sqrt(Ekin*2/m[2]),0] # E = mv^2/2
# v in terms of c
xtp, ytp, xhf, yhf, xlf, ylf, vxtp, vytp, vxhf, vyhf, vxlf, vylf = odeint(odeFunction, r + v, dt).T
v2 = [vxtp[-1],vxhf[-1],vxlf[-1]]
print('r:  '+str(r)+'\tfm')
print('m:  '+str(m)+'\tMeV/c^2')
print('v1: '+str([v[0],v[2],v[4]])+'\tc')
print('v2: '+str([v2[0],v2[1],v2[2]])+'\tc')
print('r2: '+str([xtp[-1],xhf[-1],xlf[-1]])+'\tfm')
print('t:  '+str([(xtp[-1]-r[0])/(v2[0]*3*10**8),(xhf[-1]-r[2])/(v2[1]*3*10**8),(xlf[-1]-r[4])/(v2[2]*3*10**8)])+'\tfs')
print('t:  '+str([(xtp[-1]-r[0])/(v2[0]*3*10**8)*10**(-15),(xhf[-1]-r[2])/(v2[1]*3*10**8)*10**(-15),(xlf[-1]-r[4])/(v2[2]*3*10**8)*10**(-15)])+'\ts')
print('typical convergence time: '+str((xtp[-1]-r[0])/(v2[0]*3*10**8)*10**(-15)*22.5))

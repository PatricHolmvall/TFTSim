
from __future__ import division
import numpy as np
from scipy.stats import truncnorm
from TFTSim.tftsim_utils import *
from TFTSim.tftsim import *
import copy
from time import time
from datetime import datetime
import matplotlib.pyplot as plt

from TFTSim.particles.u235 import *
from TFTSim.particles.he4 import *
from TFTSim.particles.te134 import *
from TFTSim.particles.sr96 import *
from TFTSim.particles.ce148 import *
from TFTSim.particles.ge84 import *
from TFTSim.particles.n import *


self._Z = [self._sa.tp.Z, self._sa.hf.Z, self._sa.lf.Z]
self._rad = [crudeNuclearRadius(self._sa.tp.A),
             crudeNuclearRadius(self._sa.hf.A),
             crudeNuclearRadius(self._sa.lf.A)]
self._mff = [u2m(self._sa.tp.A), u2m(self._sa.hf.A), u2m(self._sa.lf.A)]
self._minTol = 0.1

self._Q = getQValue(self._sa.fp.mEx,self._sa.pp.mEx,self._sa.tp.mEx,self._sa.hf.mEx,self._sa.lf.mEx,self._sa.lostNeutrons)

D = 19.0

xl = np.linspace(0.0,D,100)
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



from __future__ import division
import numpy as np
from scipy.stats import truncnorm
from TFTSim.tftsim_utils import *
from TFTSim.tftsim import *
import copy
from time import time
from datetime import datetime
import matplotlib.pyplot as plt
from sympy import Symbol
from sympy.solvers import nsolve
import sympy as sp
from scipy import integrate



def getLambda(x_in,r2_in,a_in,b_in):
    lam = Symbol('lam')
    return nsolve((b_in**2+lam)*x_in**2+(a_in**2+lam)*(r2_in) - (a_in**2+lam)*(b_in**2+lam), lam, 10.0)

def potentialEnergy(x_in,r2_in,a_in,b_in,q_in):
    integrand = lambda v: 0.75*q_in*(1-(x_in**2)/(a_in**2+v)+(r2_in)/(b_in**2+v))/((b_in**2+v)*np.sqrt(a_in**2+v))
    lam = getLambda(x_in,r2_in,a_in,b_in)
    print lam,
    return integrate.quad(integrand,lam,np.inf)

#print getLambda(10,20,a,b)
#print potentialEnergy(10,20,a,b,5)

a = 2.0
b = 1.0
c = b
q = 1

ke2 = 1.43996518

npts = 200
x = np.random.uniform(-10,10,npts)
y = np.ones(npts)*10
#y = np.random.uniform(-10,10,npts)
z = np.zeros(npts)
for i in range(0,npts):
    if np.sqrt(x[i]**2+y[i]**2) > getLambda(x[i],2*y[i]**2,a,b):
        print("("+str(x[i])+","+str(y[i])+")\t"),
        sol = potentialEnergy(x[i],2*y[i]**2,a,b,q)
        print("\t"+str(sol))
        z[i] = sol
    else:
        z[i] = 0
    print(str(i)+"\t"+str(z[i]))

# define grid.
xi = np.linspace(-10,10,10)
yi = np.linspace(-10,10,10)
# grid the data.
zi = griddata(x,y,z,xi,yi,interp='linear')

CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.rainbow,
                  vmax=abs(zi).max(), vmin=-abs(zi).max())
plt.colorbar() # draw colorbar
# plot data points.
plt.scatter(x,y,marker='o',c='b',s=5,zorder=10)
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title('griddata test (%d points)' % npts)
plt.show()


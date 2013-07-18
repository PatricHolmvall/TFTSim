
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
from scipy.interpolate import griddata
import scipy.interpolate



def getLambda(x_in,y_in,a_in,b_in):
    return 0.5*(x_in**2+y_in**2-a_in**2-b_in**2) + \
           np.sqrt(x_in**2*b_in**2+y_in**2*a_in**2 - a_in**2*b_in**2 + \
                   0.25*(a_in**2+b_in**2-x_in**2-y_in**2)**2)

def potentialEnergy(x_in,y_in,a_in,b_in,q_in):
    integrand = lambda v: 0.75*q_in*(1-(x_in**2)/(a_in**2+v)-(y_in)/(b_in**2+v))/(np.sqrt(a_in**2+v)*(b_in**2+v))
    lam = getLambda(x_in,y_in,a_in,b_in)
    print lam,
    return integrate.quad(integrand,lam,np.inf)

def plotEllipse(x0_in,y0_in,a_in,b_in):
    phi = np.linspace(0.0,2*np.pi,100)
    na=np.newaxis
    x_line = x0_in + a_in*np.cos(phi[:,na])
    y_line = y0_in + b_in*np.sin(phi[:,na])

    plt.plot(x_line,y_line,'r--', linewidth=3.0)

#print getLambda(10,20,a,b)
#print potentialEnergy(10,20,a,b,5)

a = 8.0
b = 5.0
c = b
q = 1

ke2 = 1.43996518

npts = 2000


x = np.random.uniform(-10,10,npts)
y = np.zeros_like(x)
for i in range(0,len(x)):
    if -a <= x[i] <= a: 
        ydir = np.random.randint(2)*2 - 1
        y[i] = ydir*np.random.uniform(b*np.sqrt(1.0-x[i]**2/a**2),10)
    else:
        y[i] = np.random.uniform(-10,10)
    
    #if -b <= y[i] <= b:
    #    if x[i] > 0 and x[i]-a < 1.5:
    #        x[i] += 1.5
    #    elif x[i] < 0 and -x[i]-a < 1.5:
    #        x[i] -= 1.5
xy = np.zeros([npts,2])

for i in range(0,len(x)):
    xy[i] = [x[i],y[i]]
#y = np.random.uniform(-10,10,npts)
z = np.zeros(npts)
for i in range(0,npts):
    print("("+str(x[i])+","+str(y[i])+")\t"),
    sol = potentialEnergy(x[i],y[i],a,b,q)
    print("\t"+str(sol))
    z[i] = sol[0]

# define grid.
xi, yi = np.linspace(x.min(),x.max(), 100), np.linspace(y.min(),y.max(), 100)
xi, yi = np.meshgrid(xi, yi)
# grid the data.
zi = griddata(xy,z,(xi,yi),method='cubic')

CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
CS = plt.contourf(xi,yi,zi,15,
                  vmax=abs(zi).max(), vmin=zi.min())
plt.colorbar() # draw colorbar
# plot data points.
"""

rbf = scipy.interpolate.Rbf(x, y, z, function='cubic')
zi = rbf(xi, yi)

plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
           extent=[x.min(), x.max(), y.min(), y.max()])
cbar = plt.colorbar()
"""
plt.scatter(x,y,marker='o',c='b',s=5,zorder=10)
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title('Ellipsoidal potential')

plotEllipse(0,0,a,b)
plt.show()

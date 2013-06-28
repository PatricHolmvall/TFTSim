
from __future__ import division
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
from sympy import Symbol
from sympy.solvers import nsolve
import sympy as sp

def solvey(D_in, x_in, E_in, Z_in, c_in, sol_guess):
    ke2 = 1.43996518
    yval = Symbol('yval')
    try:
        ySolution = nsolve(sp.sqrt((D_in-x_in)**2+yval**2)**5*Z_in[1]*(10.0*(x_in**2+yval**2)**2+c_in[1]**2*(2*x_in**2-yval**2)) + \
                           sp.sqrt(x_in**2+yval**2)**5*Z_in[2]*(10.0*((D_in-x_in)**2+yval**2)**2+c_in[2]**2*(2*(D_in-x_in)**2-yval**2)) + \
                          -10.0*(sp.sqrt(x_in**2+yval**2)*sp.sqrt((D_in-x_in)**2+yval**2))**5* \
                                (E_in/ke2 - (Z_in[1]*Z_in[2])*(1.0/D_in+(c_in[1]**2+c_in[2]**2)/(5.0*D_in**3)))/Z_in[0],
                           yval, sol_guess)
    except ValueError:
        ySolution = 0.0
    return np.float(ySolution)

def plotEllipse(x0_in,y0_in,a_in,b_in):
    phi = np.linspace(0.0,2*np.pi,100)
    na=np.newaxis
    x_line = x0_in + a_in*np.cos(phi[:,na])
    y_line = y0_in + b_in*np.sin(phi[:,na])
    plt.plot(x_line,y_line,'b-', linewidth=3.0)

def plotConfigurationContour(D_in,Q_in,Z_in,rad_in,ab_in,ec_in,figNum_in):
    xl = np.linspace(0.0,D_in,500)
    ylQ = np.zeros_like(xl)
    ylQf = np.zeros_like(xl)
    for i in range(0,len(ylQ)):
        ylQ[i] = solvey(D_in=D_in, x_in=xl[i], E_in=Q_in, Z_in=Z_in, c_in=ec_in, sol_guess=10.0)
        
        if (D_in-(ab_in[0]+ab_in[4])) < xl[i] < (ab_in[0]+ab_in[2]):
            ylQf[i] = np.max([(ab_in[3]+ab_in[1])*np.sqrt(1.0-(xl[i]/(ab_in[2]+ab_in[0]))**2),
                              (ab_in[5]+ab_in[1])*np.sqrt(1.0-((D_in-xl[i])/(ab_in[4]+ab_in[0]))**2),
                              ylQ[i]])
        elif xl[i] < (ab_in[0]+ab_in[2]) and xl[i] < (D_in-(ab_in[0]+ab_in[4])):
            ylQf[i] = np.max([(ab_in[3]+ab_in[1])*np.sqrt(1.0-(xl[i]/(ab_in[2]+ab_in[0]))**2),ylQ[i]])
        elif xl[i] > (D_in-(ab_in[0]+ab_in[4])) and xl[i] > (ab_in[0]+ab_in[2]):
            ylQf[i] = np.max([(ab_in[5]+ab_in[1])*np.sqrt(1.0-((D_in-xl[i])/(ab_in[4]+ab_in[0]))**2),ylQ[i]])
        else:
            ylQf[i] = ylQ[i]

    fig = plt.figure(figNum_in)
    ax = fig.add_subplot(111)
    
    plt.plot(xl, ylQ, 'r--', linewidth=3.0, label='E = Q')
    plt.plot(xl, ylQf, 'b-', linewidth=3.0, label='E = Q, non-overlapping radii')

def coulombEnergies(Z_in, r_in, c_in):        
    r12x = r_in[0]-r_in[2]
    r12y = r_in[1]-r_in[3]
    r13x = r_in[0]-r_in[4]
    r13y = r_in[1]-r_in[5]
    r23x = r_in[2]-r_in[4]
    r23y = r_in[3]-r_in[5]
    d12 = np.sqrt((r12x)**2 + (r12y)**2)
    d13 = np.sqrt((r13x)**2 + (r13y)**2)
    d23 = np.sqrt((r23x)**2 + (r23y)**2)
    ke2 = 1.43996518
    
    return [ke2*Z_in[0]*Z_in[1]*(1.0/d12 + \
                                 c_in[1]**2*(3.0*(r12x/d12)**2-1.0)/(10.0*d12**3)),
            ke2*Z_in[0]*Z_in[2]*(1.0/d13 + \
                                 c_in[2]**2*(3.0*(r13x/d13)**2-1.0)/(10.0*d13**3)),
            ke2*Z_in[1]*Z_in[2]*(1.0/d23 + \
                                 (c_in[1]**2+c_in[2]**2)/(5.0*d23**3) + \
                                 6.0*c_in[1]**2*c_in[2]**2/(25.0*d23**5))]

Q = 185.891
Z = [2,52,38]
A = [4,134,96]
rad = [1.2*(A[0])**(1.0/3.0),1.2*(A[1])**(1.0/3.0),1.2*(A[2])**(1.0/3.0)]
betas = [1,1,1]
ab = [rad[0],rad[0],rad[1],rad[1],rad[2],rad[2]]
ec = [0,0,0]
y=10.0
D=25.1
x=ab[0]+ab[2]
r = [0,y,-x,0,D-x,0]

for i in range(0,len(betas)):
    if not np.allclose(betas[i],1):
        # Do stuff
        ab[i*2] = rad[i]*betas[i]**(2.0/3.0)
        ab[i*2+1] = rad[i]*betas[i]**(-1.0/3.0)
        ec[i] = np.sqrt(ab[i*2]**2-ab[i*2+1]**2)

Ec = coulombEnergies(Z,r,ec)
print(Ec)
print(np.sum(Ec))

plotConfigurationContour(r[4]-r[2],Q,Z,rad,ab,ec,1)
plotEllipse(-r[2],r[1],ab[0],ab[1])
plotEllipse(0,0,ab[2],ab[3])
plotEllipse(r[4]-r[2],0,ab[4],ab[5])

plt.show()


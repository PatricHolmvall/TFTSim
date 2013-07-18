
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


def plotEllipse(x0_in,y0_in,a_in,b_in):
    phi = np.linspace(0.0,2*np.pi,100)
    na=np.newaxis
    x_line = x0_in + a_in*np.cos(phi[:,na])
    y_line = y0_in + b_in*np.sin(phi[:,na])
    plt.plot(x_line,y_line,'b-', linewidth=3.0)

def ellipseCircleOverlap(r_in,a_in,b_in,rad_in,ellipseNumber):
    xe = r_in[2]-r_in[0]
    ye = r_in[3]-r_in[1]
    plt.plot([r_in[0],r_in[2]],[r_in[1],r_in[3]],'r--')
    #if np.sqrt(xe**2+ye**2) - 1/np.sqrt(xe**2/(a_in**2*(xe**2+ye**2)) + ye**2/(b_in**2*(xe**2+ye**2))) < rad_in:
    if xe**2/(a_in+rad_in)**2 + ye**2/(b_in+rad_in)**2 <= 1:
        print("Ellipse "+str(ellipseNumber)+" and Circle overlap! "),
        print(str(xe**2/(a_in+rad_in)**2)+"+"+str(ye**2/(b_in+rad_in)**2)+"="+str(xe**2/(a_in+rad_in)**2+ye**2/(b_in+rad_in)**2)+" <= 1")
        #print(str(np.sqrt(xe**2+ye**2))+"-"+str(1/np.sqrt(xe**2/(a_in**2*(xe**2+ye**2)) + ye**2/(b_in**2*(xe**2+ye**2))))+\
        #      "="+str(np.sqrt(xe**2+ye**2)-1/np.sqrt(xe**2/(a_in**2*(xe**2+ye**2)) + ye**2/(b_in**2*(xe**2+ye**2))))+"<"+str(rad_in))

def ellipsesOverlap(r_in,a_in):
    plt.plot([r_in[0],r_in[2]],[r_in[1],r_in[3]],'g--')
    plt.scatter([r_in[0]+a_in[0],r_in[2]-a_in[2]],[0,0],c='b',s=30)
    if abs(r_in[0]-r_in[2]) <= (a_in[0]+a_in[2]):
        print("Ellipses overlap! "),
        print("|"+str(r_in[0])+"-"+str(r_in[2])+"|="+str(abs(r_in[0]-r_in[2]))+" <= "+str(a_in[0]+a_in[2]))

A=[4,134,96]
rad=[1.2*(A[0])**(1.0/3.0),1.2*(A[1])**(1.0/3.0),1.2*(A[2])**(1.0/3.0)]
ab=[1*rad[1],1*rad[1],1*rad[2],1*rad[2]]

x=0
D=ab[0]+ab[2]
y=rad[0]+ab[1]
r=[0,y,-x,0,D-x,0]

ellipseCircleOverlap(r[0:4],ab[0],ab[1],rad[0],1)
ellipseCircleOverlap(r[0:2]+r[4:6],ab[2],ab[3],rad[0],2)
ellipsesOverlap(r[2:6],ab)
plotEllipse(r[0],r[1],rad[0],rad[0])
plotEllipse(r[2],r[3],ab[0],ab[1])
plotEllipse(r[4],r[5],ab[2],ab[3])
plt.scatter([r[0],r[2],r[4]],[r[1],r[3],r[5]],c='b',s=30)

plt.show()


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

"""
Demonstrates some simple data analysis preformed on a number of simulations.
"""

from __future__ import division
import numpy as np
import scipy.interpolate
from scipy.interpolate import griddata
import pylab as pl
import matplotlib as ml
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import shelve
from TFTSim.interactions.pointparticle_coulomb import *
from TFTSim.tftsim_utils import *

simulationPaths = ["Test/2013-06-07/12.09.36/", #0
                   "Test/2013-06-07/12.33.54/", #1
                   "Test/2013-06-07/14.44.16/", #2
                   "Test/2013-06-07/17.08.04/", #3
                   "Test/2013-06-09/15.06.29/", #4  Used intial px, py both in positive direction
                   "Test/2013-06-09/16.36.53/", #5  Used initial px, py, x in random direction
                   "Test/2013-06-09/18.06.31/", #6  p = 0, x = 25% l, 75% h
                   "Test/2013-06-09/19.37.41/", #7  p = 0, x = 33% l, 66% h
                   "Test/2013-06-09/21.14.02/", #8  p = 0, x = 40% l, 60% h
                   "Test/2013-06-10/08.10.31/", #9  p = 0, x = U[0,1]
                   "Test/2013-06-10/10.34.39/", #10 p = 0, x = U[0,1], D&y = Gamma dist, ymin = 0.5
                   "Test/2013-06-10/11.49.14/", #11 p = 0, x = U[0,1], D=G(1.5,2) y=G(1.5,1), ymin = 0.05
                   "Test/2013-06-10/13.06.13/", #12 p = 0, x = U[0,1], D=G(1.5,1) y=G(1.5,1), ymin = 0.05
                   "Test/2013-06-10/13.58.01/", #13 p = 0, x = U[0,1], D=G(1.5,1) y=G(1.5,2), ymin = 0.05
                   "Test/2013-06-10/14.58.08/", #14 p = 0, x = U[0,1], D=G(1.5,2) y=G(1.5,2), ymin = 1.0
                   "Test/2013-06-10/16.00.23/", #15 p = 0, x = U[0,1], D=G(1.5,2) y=U(0.5,6)
                   "Test/2013-06-11/20.58.40/", #16 p = 0, x = U[0,1], D=G(1.5,2) y=U(0.5,6), ymin = 0
                   
                   # Generator Three
                   "Test/2013-06-12/15.03.25/", #17 D = 18
                   "Test/2013-06-12/15.40.53/", #18 D = 19
                   "Test/2013-06-12/15.59.07/", #19 D = 20
                   
                   # Generator Three version 2
                   "Test/2013-06-13/11.45.25/", #20 D = 18.1
                   "Test/2013-06-13/12.20.02/", #21 D = 18.1 higher resolution
                   "Test/2013-06-13/17.46.22/", #22 D = 18.1 introduced shelvedStaticVariables
                   "Test/2013-06-10/"
                  ]

simulations = [simulationPaths[22]]




for sim in simulations:
    sv = shelve.open("results/" + sim + 'shelvedStaticVariables.sb')
    for row in sv:
        Qval = sv[row]['Q']
        Dval = sv[row]['D']
        part1 = sv[row]['particles'][0]
        part2 = sv[row]['particles'][1]
        part3 = sv[row]['particles'][2]
        pint = sv[row]['interaction']
    sv.close()


c = 0
tot = 0
for sim in simulations:
    sv = shelve.open("results/" + sim + 'shelvedVariables.sb')
    for row in sv:
        if sv[row]['status'] == 0 and sv[row]['angle'] > 5 and sv[row]['Ekin'][0] > 7:
            c += 1
    tot += len(sv)
    sv.close()


if c == 0:
    print('No allowed data points in given data series.')
else:
    xy_forbidden = np.zeros([tot-c,2])
    xy_allowed = np.zeros([c,2])
    Ec = np.zeros(c)
    a = np.zeros(c)
    Ea = np.zeros(c)
    Ef = np.zeros(c)
    Ekin = np.zeros([c,3])
    runs = np.zeros(c)
    Ds = np.zeros(c)


    c2 = 0
    c3 = 0
    for sim in simulations:
        sv = shelve.open("results/" + sim + 'shelvedVariables.sb')
        for row in sv:
            if sv[row]['status'] == 0 and sv[row]['angle'] > 5 and sv[row]['Ekin'][0] > 7:
                Ec[c2] = np.sum(sv[row]['Ec0'])
                a[c2] = sv[row]['angle']
                Ea[c2] = sv[row]['Ekin'][0]
                Ef[c2] = np.sum(sv[row]['Ekin'][1:3])
                runs[c2] = sv[row]['ODEruns']
                xy_allowed[c2][0] = sv[row]['r0'][2]
                xy_allowed[c2][1] = sv[row]['r0'][1]
                Ds[c2] = (sv[row]['r0'][4]-sv[row]['r0'][2])
                c2 += 1
            else:
                xy_forbidden[c3][0] = sv[row]['r0'][2]
                xy_forbidden[c3][1] = sv[row]['r0'][1]
                c3 += 1
        sv.close()


energyDistribution = False
projectedEnergyDistribution = False
angularDistribution = False
xyScatterPlot = True
xyContinousPlot = True
xyDistribution = False
DDistribution = False


Zs = [part1.Z, part2.Z, part3.Z]
rads = [crudeNuclearRadius(part1.A),
        crudeNuclearRadius(part2.A),
        crudeNuclearRadius(part3.A)]

################################################################################
#                                  Ea vs Ef                                    #
################################################################################
def _plotEnergyDist(Ef_in,Ea_in,Q_in,figNum_in,nbins=10):
    H, xedges, yedges = np.histogram2d(Ef_in,Ea_in,bins=nbins)
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
    fig = plt.figure(figNum_in)
    ax = fig.add_subplot(111)
    plt.pcolormesh(xedges,yedges,Hmasked)

    maxIndex = 0
    minIndex = 0
    ymax = yedges[0]
    ymin = yedges[0]
    for i in yedges:
        if i > ymax:
            ymax = i
            maxIndex = c
        if i < ymin:
            ymin = i
            minIndex = c

    yline = np.linspace(ymax*1.1,ymin,1000)
    xline = Q_in * np.ones(len(yline)) - yline
    plt.plot(xline,yline,'r--',linewidth=5.0,label=str('Q=%1.1f' % Q_in))
    plt.title('Energy distribution')
    plt.xlabel('Ef [MeV]')
    plt.ylabel('Ea [MeV]')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.legend()

################################################################################
#                                     Ea                                       #
################################################################################
def _plotProjectedEnergyDist(E_in,figNum_in,nbins=50): 
    fig = plt.figure(figNum_in)
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(E_in, bins=nbins)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    # add a 'best fit' line for the normal PDF
    #y = mlab.normpdf( bincenters)
    l = ax.plot(bincenters, n, 'r--', linewidth=1)
    ax.set_title('Energy distribution of ternary particle')
    ax.set_xlabel('Energy [MeV]')
    ax.set_ylabel('counts')
    max = 0
    for i in range(len(n)):
        if n[i] > max:
            max = n[i]
            maxIndex = i
    plt.text(bincenters[maxIndex]+2, 0.95*n[maxIndex], str('%1.1f' % bincenters[maxIndex]),fontsize=20)

################################################################################
#                             angular distribution                             #
################################################################################
def _plotAngularDist(angles_in,figNum_in,nbins=50):
    fig = plt.figure(figNum_in)
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(angles_in, bins=nbins)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    # add a 'best fit' line for the normal PDF
    #y = mlab.normpdf( bincenters)
    l = ax.plot(bincenters, n, 'r--', linewidth=1)
    ax.set_title('Angular distribution')
    ax.set_xlabel('Angle [degrees]')
    ax.set_ylabel('Counts')

    max = 0
    for i in range(len(n)):
        if n[i] > max:
            max = n[i]
            maxIndex = i

    plt.text(bincenters[maxIndex]+2, 0.95*n[maxIndex], str('%1.1f' % bincenters[maxIndex]),fontsize=20)


################################################################################
#                   allowed / forbidden inital configurations                  #
################################################################################
def _plotConfigurationScatter(xa_in,ya_in,xf_in,yf_in,z_in,figNum_in,label_in):
    fig = plt.figure(figNum_in)
    ax = fig.add_subplot(111)
    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = np.linspace(np.min(z_in),np.max(z_in))
    norm = ml.colors.BoundaryNorm(bounds, cmap.N)
    # make the scatter
    scat = ax.scatter(xa_in,ya_in,c=z_in,marker='o',cmap=cmap,label='allowed')
    scat = ax.scatter(xf_in,yf_in,c='r',marker='s',cmap=cmap,label='forbidden')
    ax.set_title('Starting configurations of TP relative to H.')
    ax.set_xlabel('x [fm]')
    ax.set_ylabel('y [fm]')
    # create a second axes for the colorbar
    ax2 = fig.add_axes([0.90, 0.1, 0.03, 0.8])
    cb = ml.colorbar.ColorbarBase(ax2, cmap=cmap, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    cb.set_label(label_in)
    ax.legend()


################################################################################
#              allowed / forbidden inital configurations, continous            #
################################################################################
def _plotConfigurationContour(x_in,y_in,z_in,D_in,Q_in,Z_in,rad_in,pint_in,figNum_in,label_in):
    fig = plt.figure(figNum_in)
    ax = fig.add_subplot(111)
    xi, yi = np.linspace(x_in.min(), x_in.max(), 100), np.linspace(y_in.min(), y_in.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    rbf = scipy.interpolate.Rbf(x_in, y_in, z_in, function='linear')
    zi = rbf(xi, yi)

    plt.imshow(zi, vmin=z_in.min(), vmax=z_in.max(), origin='lower',
               extent=[x_in.min(), x_in.max(), y_in.min(), y_in.max()])
    plt.scatter(x_in, y_in, c=z_in,s=1)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(label_in)
    plt.title('Starting configurations of TP relative to H.')
    plt.xlabel('x [fm]')
    plt.ylabel('y [fm]')
    
    xl = np.linspace(0.0,D_in,500)
    ylQ = np.zeros_like(xl)
    ylQf = np.zeros_like(xl)
    for i in range(0,len(ylQ)):
        ylQ[i] = pint_in.solvey(D_in=D_in, x_in=xl[i], E_in=Q_in, Z_in=Z_in, sol_guess=10.0)
        
        if xl[i]<rad_in[0]+rad_in[1]:
            ylQf[i] = np.max([np.sqrt((rad_in[0]+rad_in[1])**2-xl[i]**2),ylQ[i]])
        elif xl[i]>(D_in-(rad_in[0]+rad_in[2])):
            ylQf[i] = np.max([np.sqrt((rad_in[0]+rad_in[2])**2-(D_in-xl[i])**2),ylQ[i]])
        else:
            ylQf[i] = ylQ[i]
        #print('('+str(xl[i])+','+str(ylQf[i])+')')
        
    xs = np.array([0,D_in])
    ys = np.array([0,0])
    rs = np.array([rad_in[1],rad_in[2]])
    phi = np.linspace(0.0,2*np.pi,100)
    na=np.newaxis
    x_line = xs[na,:]+rs[na,:]*np.sin(phi[:,na])
    y_line = ys[na,:]+rs[na,:]*np.cos(phi[:,na])

    plt.plot(x_line,y_line,'-', linewidth=3.0)
    
    plt.plot(xl, ylQ, 'r--', linewidth=3.0, label='E = Q')
    plt.plot(xl, ylQf, 'b-', linewidth=3.0, label='E = Q, non-overlapping radii')
    plt.text(0,0, str('HF'),fontsize=20)
    plt.text(D_in,0, str('LF'),fontsize=20)

    plt.legend()

################################################################################
#                               x-y distribution                               #
################################################################################
def _plotxyHist(x_in,y_in,figNum_in,nbins=10):
    H, xedges, yedges = np.histogram2d(x_in,y_in,bins=nbins)
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
    fig = plt.figure(figNum_in)
    ax = fig.add_subplot(111)
    plt.pcolormesh(xedges,yedges,Hmasked)
    plt.title('Starting configurations of TP relative to H')
    plt.xlabel('x [fm]')
    plt.ylabel('y [fm]')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
################################################################################
#                                D distribution                                #
################################################################################
def _plotDDistribution(D_in,figNum_in,nbins=50):
    fig2 = plt.figure(figNum_in)
    ax = fig2.add_subplot(111)
    n, bins, patches = ax.hist(D_in, bins=nbins)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    # add a 'best fit' line for the normal PDF
    #y = mlab.normpdf( bincenters)
    l = ax.plot(bincenters, n, 'r--', linewidth=1)
    ax.set_title('Start values for D')
    ax.set_xlabel('D [fm]')
    ax.set_ylabel('Counts')



figNum = 0
if c > 0:
    if energyDistribution:
        figNum += 1
        _plotEnergyDist(Ef, Ea, Qval,figNum,nbins=10)
    if projectedEnergyDistribution:
        figNum += 1
        _plotProjectedEnergyDist(Ea,figNum,nbins=50)
    if angularDistribution:
        figNum += 1
        _plotAngularDist(a,figNum,nbins=50)
    if xyScatterPlot:
        figNum += 1
        _plotConfigurationScatter(-xy_allowed[:,0],xy_allowed[:,1],
                                  -xy_forbidden[:,0],xy_forbidden[:,1],
                                  a,figNum,'Angle')
        figNum += 1
        _plotConfigurationScatter(-xy_allowed[:,0],xy_allowed[:,1],
                                  -xy_forbidden[:,0],xy_forbidden[:,1],
                                  Ea,figNum,'Ea')
    if xyContinousPlot:
        figNum += 1
        _plotConfigurationContour(-xy_allowed[:,0],xy_allowed[:,1],a,
                                  Dval,Qval,Zs,rads,pint,figNum,'Angle')
        figNum += 1
        _plotConfigurationContour(-xy_allowed[:,0],xy_allowed[:,1],Ea,Dval,
                                  Qval,Zs,rads,pint,figNum,'Ea')
    if xyDistribution:
        figNum += 1
        _plotxyHist(-xy_allowed[:,0],xy_allowed[:,1],figNum,nbins=10)
    if DDistribution:
        figNum += 1
        _plotDDistribution(Ds,figNum,nbins=50)

    if figNum > 0:
        plt.show()

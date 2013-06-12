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
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import shelve

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
                   "Test/2013-06-10/"
                  ]
simulations = [simulationPaths[19]]


angularDistribution = True
energyDistribution = True
projectedEnergyDistribution = True
DDistribution = False
xyDistribution = True
xyScatterPlot = True


c = 0
tot = 0
for sim in simulations:
    sv = shelve.open("results/" + sim + 'shelvedVariables.sb')
    for row in sv:
        if sv[row]['status'] == 0:
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
            if sv[row]['status'] == 0:
                Ec[c2] = np.sum(sv[row]['Ec0'])
                a[c2] = sv[row]['angle']
                Ea[c2] = sv[row]['Ekin'][0]
                Ef[c2] = np.sum(sv[row]['Ekin'][1:3])
                runs[c2] = sv[row]['runs']
                xy_allowed[c2][0] = sv[row]['r0'][2]
                xy_allowed[c2][1] = sv[row]['r0'][1]
                Ds[c2] = (sv[row]['r0'][4]-sv[row]['r0'][2])
                c2 += 1
            else:
                xy_forbidden[c3][0] = sv[row]['r0'][2]
                xy_forbidden[c3][1] = sv[row]['r0'][1]
                c3 += 1
            Qval = sv[row]['Q']
        sv.close()

#print("Runs per simulation [mean,std,min,max]: ["+str(np.mean(runs))+','+\
#      str(np.std(runs))+","+str(int(np.min(runs)))+","+str(int(np.max(runs)))+"]")

# scatter area
# ea vs ef 2d hist
# angle 1d hist

figNum = 1

################################################################################
#                                  Ea vs Ef                                    #
################################################################################
if energyDistribution and c > 0:
    nbins = 10
    H, xedges, yedges = np.histogram2d(Ef,Ea)#,bins=nbins)
    # H needs to be rotated and flipped
    #H = np.rot90(H)
    #H = np.flipud(H)
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
    fig1 = plt.figure(figNum)
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
    xline = Qval * np.ones(len(yline)) - yline
    plt.plot(xline,yline,'r--',linewidth=5.0,label=str('Q=%1.1f' % Qval))
    plt.title('Energy distribution')
    plt.xlabel('Ef [MeV]')
    plt.ylabel('Ea [MeV]')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.legend()
    figNum += 1

################################################################################
#                                     Ea                                       #
################################################################################
if projectedEnergyDistribution and c > 0:
    nbins = 50
    fig2 = plt.figure(figNum)
    ax = fig2.add_subplot(111)
    n, bins, patches = ax.hist(Ea, bins=nbins)
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
    figNum += 1

################################################################################
#                             angular distribution                             #
################################################################################
if angularDistribution and c > 0:
    nbins = 50
    fig2 = plt.figure(figNum)
    ax = fig2.add_subplot(111)
    n, bins, patches = ax.hist(a, bins=nbins)
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
    figNum += 1


################################################################################
#                   allowed / forbidden inital configurations                  #
################################################################################
if xyScatterPlot and c > 0:
    pl.figure(figNum)
    pl.scatter(-xy_allowed[:,0],xy_allowed[:,1],c='b',label='allowed')
    pl.scatter(-xy_forbidden[:,0],xy_forbidden[:,1],c='r',marker='s',label='forbidden')
    pl.title('Starting configurations of TP relative to H.')
    pl.xlabel('x [fm]')
    pl.ylabel('y [fm]')
    pl.legend()
    figNum += 1


################################################################################
#                               x-y distribution                               #
################################################################################
if xyDistribution and c > 0:
    nbins = 10
    H4, xedges4, yedges4 = np.histogram2d(-xy_allowed[:,0],xy_allowed[:,1])#,bins=nbins)
    Hmasked4 = np.ma.masked_where(H4==0,H4) # Mask pixels with a value of zero
    fig1 = plt.figure(figNum)
    plt.pcolormesh(xedges4,yedges4,Hmasked4)
    plt.title('Starting configurations of TP relative to H.')
    plt.xlabel('x [fm]')
    plt.ylabel('y [fm]')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    figNum += 1

################################################################################
#                                D distribution                                #
################################################################################
if DDistribution and c > 0:
    nbins = 50
    fig2 = plt.figure(figNum)
    ax = fig2.add_subplot(111)
    n, bins, patches = ax.hist(Ds, bins=nbins)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    # add a 'best fit' line for the normal PDF
    #y = mlab.normpdf( bincenters)
    l = ax.plot(bincenters, n, 'r--', linewidth=1)
    ax.set_title('Start values for D.')
    ax.set_xlabel('D [fm]')
    ax.set_ylabel('counts')
    figNum += 1


plt.show()


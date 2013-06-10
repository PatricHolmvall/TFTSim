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
Preform simple data analysis on data files generated by TRIM.

"""

from __future__ import division
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import shelve
import csv

simulationPaths = ["Ni-Al2O3/2013-06-10/13.36.33/TRIMOUT.txt", #0
                   "Ni-Al2O3/2013-06-10/13.36.33/", #1
                   "Ni-Al2O3/2013-06-10/13.36.33/", #2
                   ""
                   ]
simulations = [simulationPaths[0]]

for sim in simulations:
    print('Simulation: ' + sim)
    data = np.genfromtxt('trimData/' + sim, dtype=None, usecols=(-3,-2,-1), invalid_raise=False)
    print('coord\tmean\t\tstd')
    print('x\t%f\t%f' % (np.mean(data[:,0]),np.std(data[:,0])))
    print('y\t%f\t%f' % (np.mean(data[:,1]),np.std(data[:,1])))
    print('z\t%f\t%f' % (np.mean(data[:,2]),np.std(data[:,2])))

c = 0
data2 = np.zeros_like(data)
for row in data:
    l = np.sqrt(data[c,0]**2+data[c,1]**2+data[c,2]**2)
    data2[c,0] = data[c,0]*l
    data2[c,1] = data[c,1]*l
    data2[c,2] = data[c,2]*l
    c += 1
"""
data = np.loadtxt('trimData/' + sim, dtype={'rows': ('status',
                                                     'ion_number',
                                                     'Z',
                                                     'energy',
                                                     'depth_x',
                                                     'lateral_y',
                                                     'lateral_z',
                                                     'cos_x',
                                                     'cos_y',
                                                     'cos_z'),
                                             'formats': ('S1',
                                                        'i4',
                                                        'f4')})
"""
    #print('x\t'+str(np.mean(data[0]))+'\t'+str(np.std(data[0])))
nbins = 10
H, xedges, yedges = np.histogram2d(data2[:,1],data2[:,2])#,bins=nbins)
# H needs to be rotated and flipped
#H = np.rot90(H)
#H = np.flipud(H)
# Mask zeros
Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
fig1 = plt.figure(1)

plt.pcolormesh(xedges,yedges,Hmasked)

plt.title('Scattering distribution')
plt.xlabel('')
plt.ylabel('')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts')

plt.show()
"""
with open('trimData/' + sim, 'rb') as simFile:
    count = 0
    data = csv.reader(simFile, delimiter='\t')
    for row in data:
        if count < 10:
            print row
        count += 1
    #table = [row for row in data]
print(count)
"""
"""
c = 0
tot = 0
for sim in simulations:
    sv = shelve.open("results/" + sim + 'shelvedVariables.sb')
    for row in sv:
        if sv[row]['status'] == 0:
            c += 1
    tot += len(sv)
    sv.close()

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

################################################################################
#                                  Ea vs Ef                                    #
################################################################################
if energyDistribution:
    nbins = 10
    H, xedges, yedges = np.histogram2d(Ef,Ea)#,bins=nbins)
    # H needs to be rotated and flipped
    #H = np.rot90(H)
    #H = np.flipud(H)
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
    fig1 = plt.figure(1)
    plt.pcolormesh(xedges,yedges,Hmasked)

    maxIndex = 0
    ymax = yedges[0]
    for i in yedges:
        if i > ymax:
            ymax = i
            maxIndex = c

    yline = np.linspace(ymax*1.1,0,1000)
    xline = Qval * np.ones(len(yline)) - yline
    plt.plot(xline,yline,'r--',linewidth=5.0,label=str('Q=%1.1f' % Qval))
    plt.title('Energy distribution')
    plt.xlabel('Ef')
    plt.ylabel('Ea')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.legend()

################################################################################
#                             angular distribution                             #
################################################################################
if angularDistribution:
    nbins = 50
    fig2 = plt.figure(2)
    ax = fig2.add_subplot(111)
    n, bins, patches = ax.hist(a, bins=nbins)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    # add a 'best fit' line for the normal PDF
    #y = mlab.normpdf( bincenters)
    l = ax.plot(bincenters, n, 'r--', linewidth=1)
    ax.set_title('Angular distribution')
    ax.set_xlabel('angle [degrees]')
    ax.set_ylabel('counts')

    max = 0
    for i in range(len(n)):
        if n[i] > max:
            max = n[i]
            maxIndex = i

    plt.text(bincenters[maxIndex]+2, 0.95*n[maxIndex], str('%1.1f' % bincenters[maxIndex]),fontsize=20)


################################################################################
#                   allowed / forbidden inital configurations                  #
################################################################################
if xyScatterPlot:
    pl.figure(3)
    pl.scatter(-xy_allowed[:,0],xy_allowed[:,1],c='b',label='allowed')
    pl.scatter(-xy_forbidden[:,0],xy_forbidden[:,0],c='r',marker='s',label='forbidden')
    pl.title('Starting configurations of TP relative to H.')
    pl.xlabel('x')
    pl.ylabel('y')
    pl.legend()


################################################################################
#                               x-y distribution                               #
################################################################################
if xyDistribution:
    nbins = 10
    H4, xedges4, yedges4 = np.histogram2d(-xy_allowed[:,0],xy_allowed[:,1])#,bins=nbins)
    Hmasked4 = np.ma.masked_where(H==0,H4) # Mask pixels with a value of zero
    fig1 = plt.figure(4)
    plt.pcolormesh(xedges4,yedges4,Hmasked4)
    plt.title('Starting configurations of TP relative to H.')
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')

################################################################################
#                                D distribution                                 #
################################################################################
if DDistribution:
    nbins = 50
    fig2 = plt.figure(5)
    ax = fig2.add_subplot(111)
    n, bins, patches = ax.hist(Ds, bins=nbins)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    # add a 'best fit' line for the normal PDF
    #y = mlab.normpdf( bincenters)
    l = ax.plot(bincenters, n, 'r--', linewidth=1)
    ax.set_title('Start values for D.')
    ax.set_xlabel('D')
    ax.set_ylabel('counts')


plt.show()
"""


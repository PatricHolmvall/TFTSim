# TFTSim: Ternary Fission Trajectory Simulation in Python.
# Copyright (C) 2013 Patric Holmvall mail: patric.hol {at} gmail {dot} com
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
#from TFTSim.interactions.pointparticle_coulomb import *
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
                   "Test/2013-06-14/10.09.39/", #23 D = 18.1
                   "Test/2013-06-17/09.26.23/", #24 D = 18.1 larger sample space
                   "Test/2013-06-17/10.17.02/", #25 D = 18.5
                   "Test/2013-06-17/10.50.16/", #26 D = 19.0
                   "Test/2013-06-17/11.27.49/", #27 D = 18.0
                   "Test/2013-06-17/13.04.31/", #28 D = 18.1 higher time resolution
                   "Test/2013-06-17/14.26.53/", #29 D = 18.1 ke2 = 1.0
                   "Test/2013-06-17/16.18.46/", #30 D = 18.1 let radii overlap
                   "Test/2013-06-18/10.27.51/", #31 D = 18.1 vxtp = 1 MeV
                   "Test/2013-06-18/11.56.56/", #32 D = 18.1 vxtp = -1 MeV
                   "Test/2013-06-18/12.53.02/", #33 D = 18.1 vxtp,vytp = 0.8,0.2 MeV
                   "Test/2013-06-18/13.34.08/", #34 D = 18.1 vxtp,vytp = -0.8,0.2 MeV
                   "Test/2013-06-19/09.55.43/", #35 D = 18.1 vxtp,vytp = random direction, max 4 MeV
                   "Test/2013-06-19/10.37.49/", #36 D = 18.1 vxtp,vytp = random direction, max 4 MeV
                   "Test/2013-06-20/11.53.18/", #37 D = 18.1 Ekin accumulation test
                   "Test/2013-06-25/13.43.09/", #38 D = 18-20 Ekin accumulation test
                   "Test/2013-06-25/22.35.07/", #39 D = 18.2, 148Ce 84Ge 
                   "Test/2013-06-26/10.05.06/", #40 D = 18.2, 148Ce 84Ge, let radii overlap
                   "Test/2013-06-26/11.01.16/", #41 D = 18.2, 148Ce 84Ge, vary ekin angle and radii
                   "Test/2013-06-26/12.36.29/", #42 D = 18.2, 134Te 96Sr, vary ekin angle and radii
                   "Test/2013-06-26/14.17.33/", #43 D = 18.2, 134Te 96Sr, vary ekin angle and radii - MAAANY SAMPLES
                   "Test/2013-06-28/21.56.03/", #44 D = 20.1, ELLIPSOIDAL POTENTIAL
                   "Test/2013-07-01/11.34.44/", #45 D = 20.1, ELLIPSOIDAL POTENTIAL - beta = [1,1.5,1.5]
                   "Test/2013-07-01/12.25.57/", #46 D = 20.1, ELLIPSOIDAL POTENTIAL - beta = [1,1.5,1]
                   "Test/2013-07-01/13.44.06/", #47 D = 20.1, ELLIPSOIDAL POTENTIAL - beta = [1,1,1]
                   "Test/2013-07-01/14.35.37/", #48 D = 20.1, SPHERICAL   POTENTIAL
                   "Test/2013-07-01/16.19.39/", #49 D = 20.1, Ellipsoidal 1.5,1 - many samples in 16MeV region
                   "Test/2013-07-02/10.24.22/", #50 D = 20.1, Ellipsoidal 1.5,1 - many samples in 16MeV region
                   "Test/2013-07-02/16.28.17/", #51 D = 20.1, Ellipsoidal 1.5,1 - many samples in cylinder region
                   "Test/2013-07-09/10.31.57/", #52 D = 18.651 to 30, Ekin0 = 0, Binary Fission
                   "Test/2013-07-11/21.13.40/", #53 Test of GeneratorFive, v0 = 0 beta = [1, 1, 1]
                   "Test/2013-07-12/09.28.25/", #54 Test of GeneratorFive, v0 = 0, beta = [1, 1.5, 1]
                   "Test/2013-07-12/11.07.28/", #55 Test of GeneratorFive, v0 = 0, beta = [1, 1.25, 1]
                   "Test/2013-07-12/12.48.33/", #56 Test of GeneratorFive, v0 = 0, beta = [1, 1.3, 1]
                   "Test/2013-07-12/14.32.24/", #57 Test of GeneratorFive, v0 = 0, beta = [1, 1.35, 1]
                   "Test/2013-07-15/09.40.24/", #58 Test of GeneratorFive, v0 = 0, beta = [1, 1.4, 1]
                   "Test/2013-07-15/11.21.45/", #59 Test of GeneratorFive, v0 = 0, beta = [1, 1.45, 1]
                   "Test/2013-07-19/11.11.36/", #60 CCT: 235U -> 68Ni + 32Si + 132Sn + 2n
                   "Test/2013-07-19/14.28.54/", #61 CCT: 235U -> 68Ni + 32Si + 132Sn + 2n, triad, 10 ys
                   "Test/2013-07-19/18.29.56/", #62 CCT: 235U -> 68Ni + 32Si + 132Sn + 2n, triad, 10 ys - old timestep
                   "Test/2013-07-19/18.33.29/", #63 CCT: 235U -> 68Ni + 32Si + 132Sn + 2n, triad, 10 ys - new timestep
                   "Test/2013-07-24/16.27.29/", #64 GeneratorFive, fixed initial momenta, Ekin_limit = 13 MeV, I135, Rb95
                   "Test/2013-07-25/11.51.23/", #65 GeneratorFive, fixed initial momenta, Ekin_limit = Q-Ec, I135, Rb95
                   "Test/2013-07-25/18.21.26/", #66 GeneratorFive, gaussian initial momenta, py=py2, I135, Rb95
                   "Test/2013-07-26/09.48.55/", #67 GeneratorFive, gaussian initial momenta, py=py, I135, Rb95
                   "Test/2013-07-26/14.34.09/", #68 GeneratorFive, gaussian, py=py2, Te134, Sr96
                   "Test/2013-07-26/16.20.01/", #69 CCT + GeneratorFive, sigmax = 1, py=py2
                   "Test/2013-07-26/16.27.49/", #70 CCT + GeneratorFive, sigmax = 0.5, py=py2
                   "Test/2013-07-26/16.35.32/", #71 CCT + GeneratorFive, sigmax = 1.5, py=py2
                   "Test/2013-07-26/16.42.40/", #72 CCT + GeneratorFive, sigmax = 1.5, py=py2, Eff = min(Q-...)
                   "Test/2013-07-26/16.55.44/", #73 CCT + GeneratorFive, sigmax = 0.5, py=py2, Eff = min(Q-...)
                   "Test/2013-07-26/18.31.24/", #74 CCT + GeneratorFive, sigmax = 1.0, py=py2, Eff = min(Q-...)
                   "Test/2013-07-29/11.50.22/", #75 CCT + GeneratorFive, py=+py, sigmax=1.0, sigmay=0.25, y=sqrt
                   "Test/2013-07-29/13.35.31/", #76 CCT + GeneratorFive, py=+py, sigmax=1.0, sigmay=0.5, y=sqrt
                   "Test/2013-07-29/15.15.14/", #77 CCT + GeneratorFive, py=+py, sigmax=1.0, sigmay=0.75, y=sqrt
                   "Test/2013-07-29/17.49.36/", #78 CCT + GeneratorFive, py=+py, sigmax=1.0, sigmay=1.0, y=sqrt
                   "Test/2013-07-30/09.30.38/", #79 CCT + GeneratorFive, py=+py, sigmax=1.0, sigmay=0.25, y=gauss
                   "Test/2013-07-30/11.04.30/", #80 CCT + GeneratorFive, py=+py, sigmax=1.0, sigmay=0.5, y=gauss
                   "Test/2013-07-30/13.10.07/", #81 CCT + GeneratorFive, py=+py, sigmax=1.0, sigmay=0.75, y=gauss
                   "Test/2013-07-30/15.07.45/", #82 CCT + GeneratorFive, py=+py, sigmax=1.0, sigmay=1.0, y=gauss
                    
                   "Test/2013-08-05/14.34.50/", #83 GPU-produced

                   "1/2013-06-10/"
                  ]

simulations = [simulationPaths[83]]


for sim in simulations:
    sv = shelve.open("results/" + sim + 'shelvedStaticVariables.sb')
    for row in sv:
        fissionType = sv[row]['fissionType']
        Qval = sv[row]['Q']
        Dval = sv[row]['D']
        particles = [sv[row]['particles'][0], sv[row]['particles'][1]]
        
        if fissionType != 'BF':
            particles.append(sv[row]['particles'][2])
        
        try:
            cint = sv[row]['coulombInteraction']
        except KeyError:
            cint = sv[row]['interaction']
        try:
            nint = sv[row]['nuclearInteraction']
        except KeyError:
            nint = None 
           
        try:
            cint_type = cint.name
            if cint.name == 'ellipsoidal':
                ab = sv[row]['ab']
                ec = sv[row]['ec']
            else:
                ab = []
                for p in particles:
                    ab.extend[crudeNuclearRadius(p.A), crudeNuclearRadius(p.A)]
                ec = [0]*len(particles)
        except AttributeError:
            ab = []
            for p in particles:
                ab.extend[crudeNuclearRadius(p.A), crudeNuclearRadius(p.A)]
            ec = [0]*len(particles)
    sv.close()


c = 0
tot = 0
through = 0
for sim in simulations:
    sv = shelve.open("results/" + sim + 'shelvedVariables.sb')
    for row in sv:
        print np.shape(sv[row]['wentThrough'])
        if sv[row]['wentThrough']:
            through += 1
        if sv[row]['status'] == 0:# and 77 < sv[row]['angle'] < 87 and 0 < ((sv[row]['Ekin'][0]-16.0)**2/16.0 + (np.sum(sv[row]['Ekin'][1:3])-157.5)**2/7.5**2) < 1:# and sv[row]['angle'] > 5 and sv[row]['Ekin'][0] > 5:
            c += 1
    tot += len(sv)
    sv.close()
print(str(through)+' of '+str(tot)+' went through.')
if c == 0:
    print('No allowed data points in given data series.')
else:
    xy_forbidden = np.zeros([tot-c,2])
    xy_allowed = np.zeros([c,2])
    v0 = np.zeros([c,6])
    Ec = np.zeros(c)
    a = np.zeros(c)
    Ea = np.zeros(c)
    Ef = np.zeros(c)
    Ekin = np.zeros([c,len(particles)])
    runs = np.zeros(c)
    Ds = np.zeros(c)
    Ds_forbidden = np.zeros(tot-c)
    simNumber = np.zeros(c)
    
    c2 = 0
    c3 = 0
    for sim in simulations:
        sv = shelve.open("results/" + sim + 'shelvedVariables.sb')
        for row in sv:
            if sv[row]['status'] == 0:# and 77 < sv[row]['angle'] < 87 and 0 < ((sv[row]['Ekin'][0]-16.0)**2/16.0 + (np.sum(sv[row]['Ekin'][1:3])-157.5)**2/7.5**2) < 1:# and sv[row]['angle'] > 5 and sv[row]['Ekin'][0] > 5:
                Ec[c2] = np.sum(sv[row]['Ec0'])

                """                
                if 0 < (sv[row]['Ekin'][0]-16.0)**2/16.0 + (np.sum(sv[row]['Ekin'][1:3])-157.5)**2/7.5**2 < 1:
                    a[c2] = sv[row]['angle']
                    Ea[c2] = sv[row]['Ekin'][0]
                    Ef[c2] = np.sum(sv[row]['Ekin'][1:3])
                else:
                    a[c2] = 0.0
                    Ea[c2] = 0.0
                    Ef[c2] = 0.0
                """
                simNumber[c2] = sv[row]['simNumber']
                a[c2] = sv[row]['angle']
                Ea[c2] = sv[row]['Ekin'][0]
                if fissionType == 'BF':
                    Ef[c2] = np.sum(sv[row]['Ekin'][0:len(particles)])
                else:
                    Ef[c2] = np.sum(sv[row]['Ekin'][1:len(particles)])
                Ekin[c2] = sv[row]['Ekin']
                
                runs[c2] = sv[row]['ODEruns']
                xy_allowed[c2][0] = sv[row]['r0'][2]
                xy_allowed[c2][1] = sv[row]['r0'][1]
                v0[c2] = sv[row]['v0']
                if fissionType == 'BF':
                    Ds[c2] = (sv[row]['r0'][2])
                else:
                    Ds[c2] = (sv[row]['r0'][4]-sv[row]['r0'][2])
                c2 += 1
                
                """
                plt.figure(100)
                print(len(np.array(sv[row]['Ekins'])))
                print(len(np.linspace(0,len(np.array(sv[row]['Ekins']))*3.333,len(np.array(sv[row]['Ekins'])))))
                plt.plot(0.1*np.linspace(0,len(np.array(sv[row]['Ekins']))*3.333,len(np.array(sv[row]['Ekins']))),np.array(sv[row]['Ekins'])/sv[row]['Ekins'][-1]*100.0)
                plt.xlabel('Time 1e-21 [s]')
                plt.ylabel('Ekin/Ec0 * 100 [%]')
                """
            else:
                if fissionType == 'BF':
                    Ds_forbidden[c3] = sv[row]['r0'][2]
                else:
                    Ds_forbidden[c3] = (sv[row]['r0'][4]-sv[row]['r0'][2])
                xy_forbidden[c3][0] = sv[row]['r0'][2]
                xy_forbidden[c3][1] = sv[row]['r0'][1]
                c3 += 1
        sv.close()

print(str(c2)+' out of '+str(tot)+' runs are allowed.')
print('Ea_max: '+str(np.max(Ea)))
print('Ea_mean: '+str(np.mean(Ea)))
print('Ef_mean: '+str(np.mean(Ef)))
print('ODEruns mean: '+str(np.mean(runs)))
energyDistribution = True
projectedEnergyDistribution = True
angularDistribution = True
xyScatterPlot = False
xyContinousPlot = False
xyDistribution = True
DDistribution = True
energyAngleCorrelation = True
DvsEnergy = True
cctAnalysis = False

plotForbidden = True

Zs = []
rads = []
for p in particles:
    Zs.append(p.Z)
    rads.append(crudeNuclearRadius(p.A))

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
def _plotProjectedEnergyDist(E_in,figNum_in,title_in,nbins=50): 
    fig = plt.figure(figNum_in)
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(E_in, bins=nbins)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    # add a 'best fit' line for the normal PDF
    #y = mlab.normpdf( bincenters)
    l = ax.plot(bincenters, n, 'r--', linewidth=1)
    ax.set_title(title_in)
    ax.set_xlabel('Energy [MeV]')
    ax.set_ylabel('counts')
    max = 0
    for i in range(len(n)):
        if n[i] > max:
            max = n[i]
            maxIndex = i
    plt.text(bincenters[maxIndex]+2, 0.95*n[maxIndex], str('%1.1f MeV' % bincenters[maxIndex]),fontsize=20)

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
def _plotConfigurationScatter(xa_in,ya_in,xf_in,yf_in,z_in,figNum_in,label_in,z2,plotForbidden=True):
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
    for i in range(0,len(xa_in)):
        plt.text(xa_in[i],ya_in[i],str('%1.1f, %1.1f' % (z_in[i], z2[i])),fontsize=20)
    if plotForbidden:
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
def _plotConfigurationContour(x_in,y_in,z_in,D_in,rad_in,ab_in,cint_in,figNum_in,label_in,xl_in,ylQ_in,ylQf_in,plotShapes_in):
    
    fig = plt.figure(figNum_in)
    ax = fig.add_subplot(111)
    
    """
    if label_in == 'Angle':
        mask = ((80.0 < z_in) & (z_in < 85.0))
        z_in[~mask] = 79.0
        #idx = 80.0 < z_in < 85.0
    if label_in == 'Ea':
        mask = ((14.0 < z_in) & (z_in < 18.0))
        z_in[~mask] = 13.0
        #idx = 14.0 < z_in < 18.0
    """
    
    xi, yi = np.linspace(x_in.min(), x_in.max(), 100), np.linspace(y_in.min(), y_in.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    
    if plotShapes_in:
        idx = (xi/(ab_in[2]+rad_in[0]))**2 + (yi/(ab_in[3]+rad_in[0]))**2 < 1.0
        xi[idx] = None
        yi[idx] = None
        idx = ((D_in-xi)/(ab_in[4]+rad_in[0]))**2 + (yi/(ab_in[5]+rad_in[0]))**2 < 1.0
        xi[idx] = None
        yi[idx] = None
    
    rbf = scipy.interpolate.Rbf(x_in, y_in, z_in, function='cubic')
    
    zi = rbf(xi, yi)
    
    
    """
    plt.imshow(zi, vmin=z_in.min(), vmax=z_in.max(), origin='lower',
               extent=[x_in.min(), x_in.max(), y_in.min(), y_in.max()])
    plt.scatter(x_in, y_in, c=z_in,s=1)
    """
    CS = plt.contour(xi,yi,zi,25,linewidths=0.5,colors='k')
    CS = plt.contourf(xi,yi,zi,25,
                      vmax=zi.max(), vmin=zi.min())
    # SCATTER
    plt.scatter(x_in, y_in, c=z_in,s=1)

    cbar = plt.colorbar()
    cbar.ax.set_ylabel(label_in)

    plt.title('Starting configurations of TP relative to H.')
    plt.xlabel('x [fm]')
    plt.ylabel('y [fm]')
    
    if plotShapes_in:
        #ax.fill_between(xl, 0, ylQf_in, facecolor='white')
        plt.plot(xl, ylQ_in, 'r--', linewidth=3.0, label='E = Q')
        plt.plot(xl, ylQf_in, 'b--', linewidth=3.0, label='E = Q, non-overlapping radii')
        plotEllipse(0,0,ab_in[2],ab_in[3])
        plotEllipse(D_in,0,ab_in[4],ab_in[5])
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
################################################################################
#                        Energy-Angle Correlation of TP                        #
################################################################################
def _plotEnergyAngleCorr(a_in,Ea_in,figNum_in,nbins=10):
    H, xedges, yedges = np.histogram2d(a_in,Ea_in,bins=nbins)
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
    fig = plt.figure(figNum_in)
    ax = fig.add_subplot(111)
    plt.pcolormesh(xedges,yedges,Hmasked)

    plt.title('Energy-Angle Correlation')
    plt.xlabel('Angle [degrees]')
    plt.ylabel('Ea [MeV]')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    #plt.legend()
################################################################################
#               D versus Kinetic energy, mainly for binary fission             #
################################################################################
def _plotDvsEnergy(Ds_in,Ekin_in,figNum_in):
    fig = plt.figure(figNum_in)
    
    plt.plot(Ds_in,Ekin_in,'x')
    plt.title('D versus Ekin')
    plt.xlabel('D [fm]')
    plt.ylabel('Ekin [MeV]')
    #plt.legend()


# Sort list gotten from Shelved file: Sometimes shelve reads/stores data in a
# strange order
def sortList(list_in, simNumber_list_in):
    dummyZip = zip(simNumber_list_in, list_in)
    dummyZip.sort()
    dummyList, list_out = zip(*dummyZip)
    return list_out

figNum = 0
if c > 0:
    if energyDistribution:
        figNum += 1
        _plotEnergyDist(Ef, Ea, Qval,figNum,nbins=100)
    if projectedEnergyDistribution:
        figNum += 1
        _plotProjectedEnergyDist(Ea,figNum,'Energy distribution of ternary particle',nbins=100)
        figNum += 1
        _plotProjectedEnergyDist(Ekin[:,-2],figNum,'Energy distribution of heavy fragment',nbins=50)
        figNum += 1
        _plotProjectedEnergyDist(Ekin[:,-1],figNum,'Energy distribution of light fragment',nbins=50)
    if angularDistribution:
        figNum += 1
        _plotAngularDist(a,figNum,nbins=50)
    if xyScatterPlot:
        figNum += 1
        _plotConfigurationScatter(-xy_allowed[:,0],xy_allowed[:,1],
                                  -xy_forbidden[:,0],xy_forbidden[:,1],
                                  a,figNum,'Angle',z2=a,plotForbidden=plotForbidden)
        figNum += 1
        _plotConfigurationScatter(-xy_allowed[:,0],xy_allowed[:,1],
                                  -xy_forbidden[:,0],xy_forbidden[:,1],
                                  Ea,figNum,'Ea',z2=a,plotForbidden=plotForbidden)
        figNum += 1
        _plotConfigurationScatter(-xy_allowed[:,0],xy_allowed[:,1],
                                  -xy_forbidden[:,0],xy_forbidden[:,1],
                                  Ef,figNum,'Ef',z2=a,plotForbidden=plotForbidden)
    if xyContinousPlot:
        xl, ylQ, ylQf = getClosestConfigurationLine(Dval,500,Qval,Zs,cint,ab)
        figNum += 1
        _plotConfigurationContour(x_in=-xy_allowed[:,0],y_in=xy_allowed[:,1],
                                  z_in=a,D_in=Dval,rad_in=rads,ab_in=ab,cint_in=cint,
                                  figNum_in=figNum,label_in='Angle',
                                  xl_in=xl,ylQ_in=ylQ,ylQf_in=ylQf,
                                  plotShapes_in=True)
        figNum += 1
        _plotConfigurationContour(x_in=-xy_allowed[:,0],y_in=xy_allowed[:,1],
                                  z_in=Ea,D_in=Dval,rad_in=rads,ab_in=ab,cint_in=cint,
                                  figNum_in=figNum,label_in='Ea',
                                  xl_in=xl,ylQ_in=ylQ,ylQf_in=ylQf,
                                  plotShapes_in=True)
        """"figNum += 1
        _plotConfigurationContour(x_in=-xy_allowed[:,0],y_in=xy_allowed[:,1],
                                  z_in=(Ef+Ea),D_in=Dval,rad_in=rads,ab_in=ab,cint_in=cint,
                                  figNum_in=figNum,label_in='Ef+Ea',
                                  xl_in=xl,ylQ_in=ylQ,ylQf_in=ylQf,
                                  plotShapes_in=True)"""
    if xyDistribution:
        figNum += 1
        _plotxyHist(-xy_allowed[:,0],xy_allowed[:,1],figNum,nbins=100)
    if DDistribution:
        figNum += 1
        _plotDDistribution(Ds,figNum,nbins=50)
    if energyAngleCorrelation:
        figNum += 1
        _plotEnergyAngleCorr(a,Ea,figNum,nbins=10)
    if DvsEnergy:
        figNum += 1
        _plotDvsEnergy(sortList(Ds,simNumber),sortList(Ef,simNumber),figNum)
    
    if cctAnalysis:
        xr = np.zeros_like(xy_allowed[:,0])
        for i in range(0,len(xy_allowed[:,0])):
            xr[i] = (-xy_allowed[i,0] - (ab[0]+ab[2])) / (Ds[i] - (2*ab[0]+ab[2]+ab[4]))

        figNum += 1
        _plotConfigurationScatter(-xy_allowed[:,0],xy_allowed[:,1],
                                  -xy_forbidden[:,0],xy_forbidden[:,1],
                                  Ekin[:,-1],figNum,'E_Ni',z2=a,plotForbidden=plotForbidden)
        figNum += 1
        _plotProjectedEnergyDist(Ekin[:,-1],figNum,'Energy distribution of Ni',nbins=50)

        
        """figNum += 1
        _plotConfigurationContour(x_in=xr,y_in=Ds,
                                  z_in=Ekin[:,-1],D_in=Dval,rad_in=rads,ab_in=ab,cint_in=cint,
                                  figNum_in=figNum,label_in='E_Ni',
                                  xl_in=None,ylQ_in=None,ylQf_in=None,
                                  plotShapes_in=False)
        """
    ccts = 0
    for i in range(0,len(a)):
        if a[i] < 5:
            ccts += 1
    print("%1.1f percent were CCT (<5degrees)" % (100.0*(ccts/len(a))))
    if figNum > 0:
        plt.show()


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
import shelve

#simulationPath = "Test/2013-06-05/15.54.47/"
simulationPath = "Test/2013-06-05/16.31.32/"


sv = shelve.open("results/" + simulationPath + 'shelvedVariables.sb')
count = 0
for row in sv:
    if sv[row]['status'] == 0:
        count += 1

print(str(count)+" valid data points.")

Ec = np.zeros(count)
a = np.zeros(count)
Ea = np.zeros(count)
Ef = np.zeros(count)
Ekin = np.zeros([count,3])
runs = np.zeros(count)
c2 = 0
for row in sv:
    if sv[row]['status'] == 0:
        Ec[c2] = np.sum(sv[row]['Ec0'])
        a[c2] = sv[row]['angle']
        Ea[c2] = sv[row]['Ekin'][0]
        Ef[c2] = np.sum(sv[row]['Ekin'][1:2])
        runs[c2] = sv[row]['runs']
        c2 += 1

print("Runs [mean,std,min,max]: ["+str(np.mean(runs))+','+str(np.std(runs))+","+str(int(np.min(runs)))+","+str(int(np.max(runs)))+"]")

# scatter area
# ea vs ef 2d hist
# angle 1d hist

import numpy as np
 
nbins = 100
H, xedges, yedges = np.histogram2d(Ef,Ea)#,bins=nbins)

# H needs to be rotated and flipped
#H = np.rot90(H)
#H = np.flipud(H)
 
# Mask zeros
Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
 
# Plot 2D histogram using pcolor
fig1 = plt.figure(1)
plt.pcolormesh(xedges,yedges,Hmasked)
plt.xlabel('Ef')
plt.ylabel('Ea')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts')

"""
nbins = 100
H2, bin_edges = np.histogram(a,bins=nbins)
fig2 = plt.figure(2)
"""
plt.show()





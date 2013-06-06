import numpy as np
import pylab as pl
from sympy import Symbol
from sympy.solvers import nsolve
from sympy import sin, tan

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


def eqn(ztp,zh,zl,rh,rtp,rl,x,y):
    ke = 1.439964
    Q = 185.891
    a = (Q/ke)**2
    b = (ztp*zh)**2
    c = (ztp*zl)**2
    d = (zh*zl)**2
    A = rh + rtp - x*(2*rtp + rh + rl)
    
    dmin = Symbol('dmin')
    return nsolve(ztp*zh/((dmin*x+A)**2+y**2)**(0.5)+ztp*zl/((dmin*(1-x)-A)**2+y**2)**(0.5)+zh*zl/dmin-Q/ke, dmin,18)    



ztp = 2
zh = 52
zl = 38
rh = 6.3965
rtp = 1.98425
rl = 5.72357
xl = np.arange(0.0,1.0,0.5)
yl = np.arange(1.0,2.0,0.5)
d = np.zeros(len(xl)*len(yl))
c = 0
dmin = 0
minIndex = 0
for x in xl:
    for y in yl:
        d[c] = eqn(ztp,zh,zl,rh,rtp,rl,x,y)
        if d[c] < dmin:
            dmin = d[c]
            minIndex = c
        print c
        c += 1

print("(x,y,Dmin): ("+str(xl[minIndex])+","+str(yl[minIndex])+","+str(d[minIndex])+")")

"""
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(d[:,0],d[:,1],d[:,2], cmap=cm.jet, linewidth=0.2)
"""
fig = pl.figure(4)
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(xl,yl)
surf = ax.plot_surface(X, Y, d, cmap=cm.jet, linewidth=0.2)#, rstride=5, cstride=5, linewidth=0.2, cmap = cm.jet)#, norm = LogNorm())#, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#ax.set_zlim(0.11, 1.01)
#fig.colorbar(surf, shrink=0.5, aspect=5)
#ax.set_zlim([0,1])

plt.show()

"""
from mayavi import mlab

# Define the points in 3D space
# including color code based on Z coordinate.
pts = mlab.points3d(xl, yl, d, d)

# Triangulate based on X, Y with Delaunay 2D algorithm.
# Save resulting triangulation.
mesh = mlab.pipeline.delaunay2d(pts)

# Remove the point representation from the plot
pts.remove()

# Draw a surface based on the triangulation
surf = mlab.pipeline.surface(mesh)

# Simple plot.
mlab.xlabel("x")
mlab.ylabel("y")
mlab.zlabel("z")
mlab.show()
"""


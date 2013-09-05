from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

_y_r0 = 1.16 # fm
_y_a = 0.68 # fm
_y_as = 21.13 # MeV
_y_w = 2.3

A = [48, 132, 70]
Z = [20, 50, 28]
N = [28, 82, 42]
#A = [4, 134, 96]
#Z = [2, 52, 38]
#N = [2, 82, 58]
rad = [_y_r0*(A[0])**(1.0/3.0),
                _y_r0*(A[1])**(1.0/3.0),
                _y_r0*(A[2])**(1.0/3.0)]

_y_I1 = float(N[0]-Z[0])/float(A[0])
_y_I2 = float(N[1]-Z[1])/float(A[1])
_y_I3 = float(N[2]-Z[2])/float(A[2])
_y_zeta1 = rad[0] / _y_a
_y_zeta2 = rad[1] / _y_a
_y_zeta3 = rad[2] / _y_a
_y_g1 = _y_zeta1*np.cosh(_y_zeta1)-np.sinh(_y_zeta1)
_y_g2 = _y_zeta2*np.cosh(_y_zeta2)-np.sinh(_y_zeta2)
_y_g3 = _y_zeta3*np.cosh(_y_zeta3)-np.sinh(_y_zeta3)
_y_f1 = _y_zeta1**2 * np.sinh(_y_zeta1)
_y_f2 = _y_zeta2**2 * np.sinh(_y_zeta2)
_y_f3 = _y_zeta3**2 * np.sinh(_y_zeta3)
_y_a1 = _y_as*(1.0 - _y_w*(_y_I1**2))
_y_a2 = _y_as*(1.0 - _y_w*(_y_I2**2))
_y_a3 = _y_as*(1.0 - _y_w*(_y_I3**2))
    
YA_12 = (-4.0 * (_y_a/_y_r0)**2 * np.sqrt(_y_a1*_y_a2))
YA_13 = (-4.0 * (_y_a/_y_r0)**2 * np.sqrt(_y_a1*_y_a3))
YA_23 = (-4.0 * (_y_a/_y_r0)**2 * np.sqrt(_y_a2*_y_a3))
YB_12 = (_y_g1*_y_g2)
YB_13 = (_y_g1*_y_g3)
YB_23 = (_y_g2*_y_g3)
YC_12 = (-(_y_g1*_y_f2 + _y_g2*_y_f1))
YC_13 = (-(_y_g1*_y_f3 + _y_g3*_y_f1))
YC_23 = (-(_y_g2*_y_f3 + _y_g3*_y_f2))


def YukawaPlusForce(x12, x13, x23):
    eta_12 = x12 / _y_a
    eta_13 = x13 / _y_a
    eta_23 = x23 / _y_a
    F_12 = YA_12 * (YB_12*(eta_12 + 2.0)**2 + YC_12*(eta_12 + 1.0)) * np.exp(-eta_12) / (_y_a * (eta_12)**2)
    F_13 = YA_13 * (YB_13*(eta_13 + 2.0)**2 + YC_13*(eta_13 + 1.0)) * np.exp(-eta_13) / (_y_a * (eta_13)**2)
    F_23 = YA_23 * (YB_23*(eta_23 + 2.0)**2 + YC_23*(eta_23 + 1.0)) * np.exp(-eta_23) / (_y_a * (eta_23)**2)
    return F_12, F_13, F_23

def YukawaPlusPotential(x12, x13, x23):
    eta_12 = x12 / _y_a
    eta_13 = x13 / _y_a
    eta_23 = x23 / _y_a
    P_12 = YA_12 * (YB_12*(4.0 + eta_12) + YC_12) * np.exp(-eta_12) / eta_12
    P_13 = YA_13 * (YB_13*(4.0 + eta_13) + YC_13) * np.exp(-eta_13) / eta_13
    P_23 = YA_23 * (YB_23*(4.0 + eta_23) + YC_23) * np.exp(-eta_23) / eta_23
    return P_12, P_13, P_23

ke2 = 1.43996158

def CoulombForce(x12, x13, x23):
    F_12 = ke2 * Z[0] * Z[1] / x12**2
    F_13 = ke2 * Z[0] * Z[2] / x13**2
    F_23 = ke2 * Z[1] * Z[2] / x23**2
    return F_12, F_13, F_23

def CoulombPotential(x12, x13, x23):
    E_12 = ke2 * Z[0] * Z[1] / x12
    E_13 = ke2 * Z[0] * Z[2] / x13
    E_23 = ke2 * Z[1] * Z[2] / x23
    return E_12, E_13, E_23

num = 10000
xs = np.linspace(0.0, 100.0, num)
fs1 = np.zeros(num)
fs2 = np.zeros(num)
fs3 = np.zeros(num)
f1 = np.zeros(num)
f2 = np.zeros(num)
f3 = np.zeros(num)
ps1 = np.zeros(num)
ps2 = np.zeros(num)
ps3 = np.zeros(num)
p1 = np.zeros(num)
p2 = np.zeros(num)
p3 = np.zeros(num)

for i in range(0,num):
    fs1[i], fs2[i], fs3[i] = YukawaPlusForce(xs[i] + rad[0] + rad[1],
                                             xs[i] + rad[0] + rad[2],
                                             xs[i] + rad[1] + rad[2])
    ps1[i], ps2[i], ps3[i] = YukawaPlusPotential(xs[i] + rad[0] + rad[1],
                                                 xs[i] + rad[0] + rad[2],
                                                 xs[i] + rad[1] + rad[2])
    f1[i], f2[i], f3[i] = CoulombForce(xs[i] + rad[0] + rad[1],
                                       xs[i] + rad[0] + rad[2],
                                       xs[i] + rad[1] + rad[2])
    p1[i], p2[i], p3[i] = CoulombPotential(xs[i] + rad[0] + rad[1],
                                           xs[i] + rad[0] + rad[2],
                                           xs[i] + rad[1] + rad[2])

plt.figure(1)
plt.plot(xs,fs1,'r--',lw=3.0,label=r'Yukawa: $^{48}$Ca + $^{132}$Sn')
plt.plot(xs,fs2,'g--',lw=3.0,label=r'Yukawa: $^{48}$Ca + $^{70}$Ni')
plt.plot(xs,fs3,'b--',lw=3.0,label=r'Yukawa: $^{132}$Sn + $^{70}$Ni')
plt.plot(xs,f1,'r:',lw=3.0,label=r'Coulomb: $^{48}$Ca + $^{132}$Sn')
plt.plot(xs,f2,'g:',lw=3.0,label=r'Coulomb: $^{48}$Ca + $^{70}$Ni')
plt.plot(xs,f3,'b:',lw=3.0,label=r'Coulomb: $^{132}$Sn + $^{70}$Ni')
plt.plot(xs,(fs1+f1),'r-',lw=3.0,label=r'Both: $^{48}$Ca + $^{132}$Sn')
plt.plot(xs,(fs2+f2),'g-',lw=3.0,label=r'Both: $^{48}$Ca + $^{70}$Ni')
plt.plot(xs,(fs3+f3),'b-',lw=3.0,label=r'Both: $^{132}$Sn + $^{70}$Ni')
plt.title('Yukawa Plus Force')
plt.xlabel('Tip Distance [fm]')
plt.ylabel('Energy [MeV/fm]')
plt.legend(loc=4)

plt.figure(2)
plt.plot(xs,ps1,'r--',lw=3.0,label=r'Yukawa: $^{48}$Ca + $^{132}$Sn')
plt.plot(xs,ps2,'g--',lw=3.0,label=r'Yukawa: $^{48}$Ca + $^{70}$Ni')
plt.plot(xs,ps3,'b--',lw=3.0,label=r'Yukawa: $^{132}$Sn + $^{70}$Ni')
plt.plot(xs,p1,'r:',lw=3.0,label=r'Yukawa: $^{48}$Ca + $^{132}$Sn')
plt.plot(xs,p2,'g:',lw=3.0,label=r'Yukawa: $^{48}$Ca + $^{70}$Ni')
plt.plot(xs,p3,'b:',lw=3.0,label=r'Yukawa: $^{132}$Sn + $^{70}$Ni')
plt.plot(xs,(ps1+p1),'r-',lw=3.0,label=r'Both: $^{48}$Ca + $^{132}$Sn')
plt.plot(xs,(ps2+p2),'g-',lw=3.0,label=r'Both: $^{48}$Ca + $^{70}$Ni')
plt.plot(xs,(ps3+p3),'b-',lw=3.0,label=r'Both: $^{132}$Sn + $^{70}$Ni')
plt.title('Yukawa Plus Potential Energy')
plt.xlabel('Tip Distance [fm]')
plt.ylabel('Potential [MeV]')
plt.legend(loc=4)

plt.show()



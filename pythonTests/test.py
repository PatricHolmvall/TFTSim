
from __future__ import division
import numpy as np
from sympy import Symbol
from sympy.solvers import nsolve
import sympy as sp

def getEllipsoidAxes(betas_in,rad_in):
    """
    Returns the semimajor (a) and semiminor (b) axes of an ellipsoid, given
    their ratio beta and the radius of the sphere when a=b. The assumption that
    the volume is constant is used, ie r^3 = a*b^2.
    
    :type betas_in: list of floats
    :param betas_in: Ratio between semimajor and semiminor axis: beta = a/b.

    :type rad_in: list of floats
    :param rad_in: Radii of the sphere when semimajor = semiminor axis (a=b).

    :rtype: list of floats
    :returns: List of Semimajor (a) and semiminor (b) axes: [a1,b1,a2,b2,...]
              and list of difference ec = a^2-b^2: [ec1,ec2,...].
    """
    
    ab_out = []
    for r in rad_in:
        ab_out.extend([r,r])
    
    ec_out = [0]*len(betas_in)
    
    for i in range(0,len(betas_in)):
        if not np.allclose(betas_in[i],1):
            ab_out[i*2] = rad_in[i]*betas_in[i]**(2.0/3.0)
            ab_out[i*2+1] = rad_in[i]*betas_in[i]**(-1.0/3.0)
            ec_out[i] = np.sqrt(ab_out[i*2]**2-ab_out[i*2+1]**2)
    
    return ab_out, ec_out

def crudeNuclearRadius(A_in, r0=1.25):
    """
    Calculate the nuclear radius of a particle of atomic mass number A through
    the crude approximation r = r0*A^(1/3), where [r0] = fm.
    
    :type A_in: int
    :parameter A_in: Atomic mass number.
    
    :type r0: float
    :parameter r0: Radius coefficient that you might want to vary as A varies
                   to get the correct fit.
    
    :rtype: float
    :returns: Nuclear radius in fm.
    """
    
    #if A > 40:
    #    r0 = 1.16
    #else:
    #    r0 = 1.22
    #
    # r0 = 0.94 + 32.0/(Z_in**2 + 200.0)
    #4*Pi*0.95*(1-1.7826*((134-2*52)/134)^2)*((1-(1/(1.2*(134)^(1/3)))^2)^(-1)+(1-(1/(1.2*(96)^(1/3)))^2)^(-1))*(-4.41*e^(-9/0.7176))
    
    return r0 * (np.float(A_in)**(1.0/3.0))

def solveDwhenTPonAxis(xr_in, E_in, Z_in, ec_in, sol_guess):
    """
    Get D for given x, Energy, particle Z and particle radii, when ternary
    particle is on axis.
    
    :type xr_in: float
    :param xr_in: relative x displacement of ternary particle between heavy and
                 light fission fragment.
    
    :type y_in: float
    :param y_in: y displacement between ternary particle and fission axis.
    
    :type E_in: float
    :param E_in: Energy in MeV.
    
    :type Z_in: list of ints
    :param Z_in: Particle proton numbers [Z1, Z2, Z3].

    :type sol_guess: float
    :param sol_guess: Initial guess for solution.
    
    :rtype: float
    :returns: D for current equation.
    """
    
    ke2 = 1.43996518
    
    Dval = Symbol('Dval')
    return np.float(nsolve(Dval**5*(Z_in[0]*Z_in[1]*(1.0-xr_in)**2+\
                                    Z_in[0]*Z_in[2]*(1.0-xr_in)**3/xr_in+\
                                    Z_in[1]*Z_in[2]*(1.0-xr_in)**3)+\
                           Dval**3*(Z_in[0]*Z_in[1]*ec_in[1]**2/5.0+\
                                    Z_in[0]*Z_in[2]*ec_in[2]**2/5.0*\
                                                    (1.0-xr_in)**3+\
                                    Z_in[1]*Z_in[2]*(ec_in[1]**2+
                                                     ec_in[2]**2)/5.0*\
                                                    (1.0-xr_in)**3)+\
                           -Dval**6*(E_in/ke2)*(1-xr_in)**3,
                           Dval, sol_guess))

betas = [1,1,1]
Z = [2,52,38]
A = [4,134,96]
rad = [crudeNuclearRadius(A[0]),crudeNuclearRadius(A[1]),crudeNuclearRadius(A[2])]
ab, ec = getEllipsoidAxes(betas,rad)

dr = 1.0/(np.sqrt(float(Z[1])/float(Z[2])) + 1.0)
xr = 1.0-dr
E_solve = 155.5 + 15.9 - 5.0 - 13.0
dE_solve = 2*0.8 + 2*0.2

D = solveDwhenTPonAxis(xr_in=xr, E_in=E_solve, Z_in=Z, ec_in=ec, sol_guess=21.0)
print('D = '+str(D))
print('d = '+str(dr*D))


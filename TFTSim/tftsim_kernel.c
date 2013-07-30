// TFTSim: Ternary Fission Trajectory Simulation in Python.
// Copyright (C) 2013 Patric Holmvall mail: patric.hol {at} gmail {dot} com
//
// TFTSim is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TFTSim is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with TFTSim.  If not, see <http://www.gnu.org/licenses/>.

//##############################################################################
//#                                 Defines                                    #
//##############################################################################
//Description: This is a placeholder for defines used in the program.

%(defines)s

#ifdef ENABLE_DOUBLE
    // Check that pragmas for 64bit actually exists
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable

    // Set float precision to double
    #define FLOAT_TYPE double
    #define FLOAT_TYPE_V double2
    
    //#ifdef cl_khr_fp64
    //    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    //#elif defined(cl_amd_fp64)
    //    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    //#else
    //    #error "Double precision floating point not supported by OpenCL implementation."
    //#endif
#else
    // Set float precision to single
    #define FLOAT_TYPE float
    #define FLOAT_TYPE_V float2
#endif


//##############################################################################
//#                               RUNGE-KUTTA 4                                #
//##############################################################################
inline FLOAT_TYPE rk4(FLOAT_TYPE t, FLOAT_TYPE y)
{
    FLOAT_TYPE k1 = dt * rk4_f(t, y);
    FLOAT_TYPE k2 = dt * rk4_f((t + 0.5 * %(dt)s), (y + 0.5 * %(dt)s * k1));
    FLOAT_TYPE k3 = dt * rk4_f((t + 0.5 * %(dt)s), (y + 0.5 * %(dt)s * k2));
    FLOAT_TYPE k4 = dt * rk4_f((t + %(dt)s), (y + %(dt)s * k3));
    
	return y + H*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0;
}

inline FLOAT_TYPE rk4_f(FLOAT_TYPE t, FLOAT_TYPE y)
{
    
}

//##############################################################################
//#                                   VERLET                                   #
//##############################################################################

//##############################################################################
//#                              SIMPLECTIC EULER                              #
//##############################################################################

//##############################################################################
//#                             ODE SOLVER WRAPPER                             #
//##############################################################################
inline FLOAT_TYPE *odeSolve(r_solve[], v_solve[], dt)
{
    FLOAT_TYPE accel;
    accel = ODE_SOLVER (r_solve);
    return v_solve, accel;
}

//##############################################################################
//#                                KERNEL CODE                                 #
//##############################################################################
//Description: The following code is the TFTSim kernel, which is mainly and ODE
//             solver.
__kernel void
gpuODEsolver (__global FLOAT_TYPE_V *r,
              __global FLOAT_TYPE_V *v,
              __global int *status,
              __global float *errorSizeODE
             )
{
    uint threadId = get_global_id(0) + get_global_id(1) * get_global_size(0);
    // Download variables to local memory
    __local FLOAT_TYPE r_local [6];
    __local FLOAT_TYPE v_local [6];
    r_local[0] = r[threadId + 0];
    r_local[1] = r[threadId + 1];
    r_local[2] = r[threadId + 2];
    r_local[3] = r[threadId + 3];
    r_local[4] = r[threadId + 4];
    r_local[5] = r[threadId + 5];
    v_local[0] = v[threadId + 0];
    v_local[1] = v[threadId + 1];
    v_local[2] = v[threadId + 2];
    v_local[3] = v[threadId + 3];
    v_local[4] = v[threadId + 4];
    v_local[5] = v[threadId + 5];
    
    FLOAT_TYPE calc_error;
    int ode_steps;
    ode_steps = 10000;
    calc_error = 0.0;
    // Solve ODE
    for (int i = 0; i < ode_steps; i++)
    {
        r_local, v_local = odeSolve(r_local, v_local, dt);
    }
    
    // Upload variables to global memory
    
}


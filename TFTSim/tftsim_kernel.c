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

//
//
//
//##############################################################################
//#                           COULOMB ACCELERATION                             #
//##############################################################################
inline FLOAT_TYPE* coulombAcceleration(FLOAT_TYPE r_in[6], FLOAT_TYPE a_in[6])
{
    FLOAT_TYPE r12x = r_in[0]-r_in[2];
    FLOAT_TYPE r12y = r_in[1]-r_in[3];
    FLOAT_TYPE r13x = r_in[0]-r_in[4];
    FLOAT_TYPE r13y = r_in[1]-r_in[5];
    FLOAT_TYPE r23x = r_in[2]-r_in[4];
    FLOAT_TYPE r23y = r_in[3]-r_in[5];


#ifdef FULL_SPHERICAL
    FLOAT_TYPE d12i = 1.0/sqrt((r12x)*(r12x) + (r12y)*(r12y));
    FLOAT_TYPE d13i = 1.0/sqrt((r13x)*(r13x) + (r13y)*(r13y));
    FLOAT_TYPE d23i = 1.0/sqrt((r23x)*(r23x) + (r23y)*(r23y));

    FLOAT_TYPE F12r = %(Q12)s*((d12i*d12i*d12i));
    FLOAT_TYPE F12x = r12x*F12r;
    FLOAT_TYPE F12y = r12y*F12r;

    FLOAT_TYPE F13r = %(Q13)s*((d13i*d13i*d13i));
    FLOAT_TYPE F13x = r13x*F13r;
    FLOAT_TYPE F13y = r13y*F13r;

    FLOAT_TYPE F23r = %(Q23)s*((d23i*d23i*d23i));
    FLOAT_TYPE F23x = r23x*F23r;
    FLOAT_TYPE F23y = r23y*F23r;
#else
    /*
    FLOAT_TYPE d12 = sqrt((r12x)*(r12x) + (r12y)*(r12y));
    FLOAT_TYPE d13 = sqrt((r13x)*(r13x) + (r13y)*(r13y));
    FLOAT_TYPE d23 = sqrt((r23x)*(r23x) + (r23y)*(r23y));

    FLOAT_TYPE F12r = %(Q12)s*(1.0/(d12*d12) +
                               3.0*%(ec2_2)s * (3.0*(r12x*r12x/(d12*d12))-1.0) / (10.0*(d12*d12*d12*d12)));
    FLOAT_TYPE F12t = %(Q12)s*(3.0*%(ec2_2)s*r12x*r12y) / (5.0*(d12*d12*d12*d12*d12*d12));
    FLOAT_TYPE F12x = r12x*F12r/d12 + r12y*F12t;
    FLOAT_TYPE F12y = r12y*F12r/d12 + r12x*F12t;

    FLOAT_TYPE F13r = %(Q13)s*(1.0/(d13*d13) +
                               3.0*%(ec2_3)s * (3.0*(r13x*r13x/(d13*d13))-1.0) / (10.0*(d13*d13*d13*d13)));
    FLOAT_TYPE F13t = %(Q13)s*(3.0*%(ec2_3)s*r13x*r13y) / (5.0*(d13*d13*d13*d13*d13*d13));
    FLOAT_TYPE F13x = r13x*F13r/d13 + r13y*F13t;
    FLOAT_TYPE F13y = r13y*F13r/d13 + r13x*F13t;

    FLOAT_TYPE F23r = %(Q23)s*(1.0/(d23*d23) +
                               3.0*(%(ec2_2)s+%(ec2_3)s)/(5.0*(d23*d23*d23*d23)) +
                               6.0*(%(ec2_2)s*%(ec2_3)s)/(5.0*(d23*d23*d23*d23*d23*d23)));
    FLOAT_TYPE F23x = r23x*F23r/d23;
    FLOAT_TYPE F23y = r23y*F23r/d23;
    */
    FLOAT_TYPE d12sq = ((r12x)*(r12x) + (r12y)*(r12y));
    FLOAT_TYPE d13sq = ((r13x)*(r13x) + (r13y)*(r13y));
    FLOAT_TYPE d23sq = ((r23x)*(r23x) + (r23y)*(r23y));

    FLOAT_TYPE F12r = %(Q12)s*(1.0/(d12sq) +
                               %(ec2_2)s * (0.9*(r12x*r12x/(d12sq))-0.3) / (d12sq*d12sq));
    FLOAT_TYPE F12t = %(Q12)s*(0.6*%(ec2_2)s*r12x*r12y) / (d12sq*d12sq*d12sq);
    FLOAT_TYPE F12x = r12x*F12r/sqrt(d12sq) + r12y*F12t;
    FLOAT_TYPE F12y = r12y*F12r/sqrt(d12sq) + r12x*F12t;

    FLOAT_TYPE F13r = %(Q13)s*(1.0/(d13sq) +
                               %(ec2_3)s * (0.9*(r13x*r13x/(d13sq))-0.3) / (d13sq*d13sq));
    FLOAT_TYPE F13t = %(Q13)s*(0.6*%(ec2_3)s*r13x*r13y) / (d13sq*d13sq*d13sq);
    FLOAT_TYPE F13x = r13x*F13r/sqrt(d13sq) + r13y*F13t;
    FLOAT_TYPE F13y = r13y*F13r/sqrt(d13sq) + r13x*F13t;

    FLOAT_TYPE F23r = %(Q23)s*(1.0/(d23sq) +
                               0.6*(%(ec2_2)s+%(ec2_3)s)/(d23sq*d23sq) +
                               1.2*(%(ec2_2)s*%(ec2_3)s)/(d23sq*d23sq*d23sq));
    FLOAT_TYPE F23x = r23x*F23r/sqrt(d23sq);
    FLOAT_TYPE F23y = r23y*F23r/sqrt(d23sq);
#endif

    a_in[0] = (F12x + F13x) * %(m1i)s;
    a_in[1] = (F12y + F13y) * %(m1i)s;
    a_in[2] = (-F12x + F23x) * %(m2i)s;
    a_in[3] = (-F12y + F23y) * %(m2i)s;
    a_in[4] = (-F13x - F23x) * %(m3i)s;
    a_in[5] = (-F13y - F23y) * %(m3i)s;
    
    return a_in;
}

#ifdef COLLISION_CHECK
//##############################################################################
//#                              COLLISION CHECK                               #
//##############################################################################
inline bool collisionCheck(FLOAT_TYPE r_in[6])
{
    if( (r_in[2]-r_in[0])*(r_in[2]-r_in[0]) / ((%(ab2x)s+%(rad1)s)*(%(ab2x)s+%(rad1)s)) +
        (r_in[3]-r_in[1])*(r_in[3]-r_in[1]) / ((%(ab2y)s+%(rad1)s)*(%(ab2y)s+%(rad1)s)) < 1.0)
    {
        return true;
    }
    
    if( (r_in[2]-r_in[0])*(r_in[2]-r_in[0]) / ((%(ab3x)s+%(rad1)s)*(%(ab3x)s+%(rad1)s)) +
        (r_in[3]-r_in[1])*(r_in[3]-r_in[1]) / ((%(ab3y)s+%(rad1)s)*(%(ab3y)s+%(rad1)s)) < 1.0)
    {
        return true;
    }
    
    return false;
}
#endif

//##############################################################################
//#                               RUNGE-KUTTA 4                                #
//##############################################################################

//##############################################################################
//#                                   VERLET                                   #
//##############################################################################

//##############################################################################
//#                              SIMPLECTIC EULER                              #
//##############################################################################

//##############################################################################
//#                                KERNEL CODE                                 #
//##############################################################################
//Description: The following code is the TFTSim kernel, which is mainly and ODE
//             solver.
__kernel void
gpuODEsolver (__global FLOAT_TYPE *r
             ,__global FLOAT_TYPE *v
             ,__global int *status
              //,__global float *errorSizeODE
#ifdef SAVE_TRAJECTORIES
             ,__global FLOAT_TYPE *trajectories
#endif
             )
{
    uint threadId = get_global_id(0) + get_global_id(1) * get_global_size(0);
    // Download variables to local memory
    FLOAT_TYPE r_local [6];
    FLOAT_TYPE v_local [6];
    r_local[0] = r[threadId*6 + 0];
    r_local[1] = r[threadId*6 + 1];
    r_local[2] = r[threadId*6 + 2];
    r_local[3] = r[threadId*6 + 3];
    r_local[4] = r[threadId*6 + 4];
    r_local[5] = r[threadId*6 + 5];

    v_local[0] = v[threadId*6 + 0];
    v_local[1] = v[threadId*6 + 1];
    v_local[2] = v[threadId*6 + 2];
    v_local[3] = v[threadId*6 + 3];
    v_local[4] = v[threadId*6 + 4];
    v_local[5] = v[threadId*6 + 5];

    FLOAT_TYPE calc_error;
    calc_error = 0.0;
    FLOAT_TYPE r2[6]  = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    FLOAT_TYPE r3[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    FLOAT_TYPE r4[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    //FLOAT_TYPE v2[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    //FLOAT_TYPE v3[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    //FLOAT_TYPE v4[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    FLOAT_TYPE a1[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    FLOAT_TYPE a2[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    FLOAT_TYPE a3[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    FLOAT_TYPE a4[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    // Solve ODE
    for (int i = 0; i < %(odeSteps)s; i++)
    {
        coulombAcceleration(r_local, a1);
        for(int j = 0; j < 6; j++)
        {
            //v2[j] = v_local[j] + 0.5 * %(dt)s * a1[j];
            //r2[j] = r_local[j] + 0.5 * %(dt)s * v2[j];
            r2[j] = r_local[j] + 0.5 * %(dt)s * (v_local[j] + 0.5 * %(dt)s * a1[j]);
        }
        
        coulombAcceleration(r2, a2);
        for(int j = 0; j < 6; j++)
        {
            //v3[j] = v_local[j] + 0.5 * %(dt)s * a2[j];
            //r3[j] = r_local[j] + 0.5 * %(dt)s * v3[j];
            r3[j] = r_local[j] + 0.5 * %(dt)s * (v_local[j] + 0.5 * %(dt)s * a2[j]);
        }
        
        coulombAcceleration(r3, a3);
        for(int j = 0; j < 6; j++)
        {
            //v4[j] = v_local[j] + %(dt)s * a3[j];
            //r4[j] = r_local[j] + 0.5 * %(dt)s * v4[j];
            r4[j] = r_local[j] + %(dt)s * (v_local[j] + %(dt)s * a3[j]);
        }
        
        coulombAcceleration(r4, a4);
        
        for(int j = 0; j < 6; j++)
        {
            //r_local[j] = r_local[j] + %(dt)s * (v_local[j] + 2.0*v2[j] + 2.0*v3[j] + v4[j]) / 6.0;
            r_local[j] = r_local[j] + %(dt)s * v_local[j] + (%(dt)s * %(dt)s) * (a1[j] + a2[j] + a3[j]) / 6.0;
            v_local[j] = v_local[j] + %(dt)s * (a1[j] + 2.0*a2[j] + 2.0*a3[j] + a4[j]) / 6.0;

#ifdef SAVE_TRAJECTORIES
            if(i < %(trajectorySaveSize)s)
            {
                trajectories[threadId*6*%(trajectorySaveSize)s + i + j*%(trajectorySaveSize)s] = r_local[j];
            }
#endif
        }
#ifdef COLLISION_CHECK
            if(collisionCheck(r_local))
            {
                status[threadId] = 1;
            }
#endif
    }
    // Upload variables to global memory
    r[threadId*6 + 0] = r_local[0];
    r[threadId*6 + 1] = r_local[1];
    r[threadId*6 + 2] = r_local[2];
    r[threadId*6 + 3] = r_local[3];
    r[threadId*6 + 4] = r_local[4];
    r[threadId*6 + 5] = r_local[5];
    
    v[threadId*6 + 0] = v_local[0];
    v[threadId*6 + 1] = v_local[1];
    v[threadId*6 + 2] = v_local[2];
    v[threadId*6 + 3] = v_local[3];
    v[threadId*6 + 4] = v_local[4];
    v[threadId*6 + 5] = v_local[5];
    
}


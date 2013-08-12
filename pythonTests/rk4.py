
            
            ####################################################################
            #                           RK4 method 1                           #
            ####################################################################
            """
            xplot = np.zeros([1000,6])
            DT = 1.0
            tajm = time()
            vout = np.array(v)
            xout = np.array(r)
            for i in range(0,1000):
                if(i%100 == 0):
                    print i
                v1 = vout
                x1 = xout
                a1 = np.array(self._cint.accelerations(Z_in = self._Z,
                                              r_in = x1,
                                              m_in = self._mff,
                                              fissionType_in=self._fissionType))
                v2 = v1 + DT*0.5*a1
                x2 = x1 + DT*0.5*v2
                a2 = np.array(self._cint.accelerations(Z_in = self._Z,
                                              r_in = x2,
                                              m_in = self._mff,
                                              fissionType_in=self._fissionType))
                v3 = v1 + DT*0.5*a2
                x3 = x1 + DT*0.5*v3
                a3 = np.array(self._cint.accelerations(Z_in = self._Z,
                                              r_in = x3,
                                              m_in = self._mff,
                                              fissionType_in=self._fissionType))
                v4 = v1 + DT*a3
                x4 = x1 + DT*v4
                a4 = np.array(self._cint.accelerations(Z_in = self._Z,
                                              r_in = x4,
                                              m_in = self._mff,
                                              fissionType_in=self._fissionType))
                vout = v1 + DT*(a1 + 2.0*a2 + 2.0*a3 + a4)/6.0
                xout = x1 + DT*(v1 + 2.0*v2 + 2.0*v3 + v4)/6.0
                xplot[i] = xout
            tajm2 = time()
            plt.plot(xplot[:,0],xplot[:,1],'b--',linewidth=3.0)
            #plt.plot(xplot[:,2],xplot[:,3],'b--',linewidth=3.0)
            #plt.plot(xplot[:,4],xplot[:,5],'b--',linewidth=3.0)
            
            tajm3 = time()
            """

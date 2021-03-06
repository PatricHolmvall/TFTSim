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

from __future__ import division
from TFTSim.tftsim_utils import *
import sys
import numpy as np
from scipy.integrate import odeint
from time import time
from time import sleep
from datetime import datetime
from collections import defaultdict
import commands
import os
import copy
import pickle
import shelve
import pylab as pl
import matplotlib.pyplot as plt
import math
from itertools import compress

# Fancy Legend transparency:
#leg = ax.legend(loc='best', fancybox=True)
#leg.get_frame().set_alpha(0.5)

class TFTSimAnalysis:
    """
    Do analysis of TFTSim data.
    """

    def __init__(self, simulationPath, verbose=False, oldDataAnalysis=False):
        
        self._simulationPath = simulationPath
        self._verbose = verbose
        
        
    def openShelvedVariables(self):
        """
        """
        # Check that file exists
        if self._simulationPath == "" or not os.path.isfile(self._simulationPath + "shelvedVariables.sb"):
            print self._simulationPath
            raise ValueError("simulationPath must be an existing and valid .sb "
                             "file.")
        # Unshelve
        print('Starting unshelve of '+self._simulationPath+'shelvedVariables.sb.')
        shelveStart = time()
        sv = shelve.open(self._simulationPath + "shelvedVariables.sb")
        self._simData0 = {'simName': sv['0']['simName'],
                          'simulations': sv['0']['simulations'],
                          'fissionType': sv['0']['fissionType'],
                          'Q': sv['0']['Q'],
                          'D': sv['0']['D'],
                          'r': np.array(sv['0']['r']).T,
                          'v': np.array(sv['0']['v']).T,
                          'r0': np.array(sv['0']['r0']).T,
                          'v0': np.array(sv['0']['v0']).T,
                          'TXE': sv['0']['TXE'],
                          'Ec0': np.array(sv['0']['Ec0']).T,
                          'Ekin0': np.array(sv['0']['Ekin0']).T,
                          'angle': sv['0']['angle'],
                          'Ec': np.array(sv['0']['Ec']).T,
                          'Ekin': np.array(sv['0']['Ekin']).T,
                          'ODEruns': sv['0']['ODEruns'],
                          'status': sv['0']['status'],
                          'error': sv['0']['error'],
                          'wentThrough': sv['0']['wentThrough'],
                          'Ekins': sv['0']['Ekins'],
                          'particles': sv['0']['particles'],
                          'coulombInteraction': sv['0']['coulombInteraction'],
                          'nuclearInteraction': sv['0']['nuclearInteraction'],
                          'D0': sv['0']['D0'],
                          'ab': sv['0']['ab'],
                          'ec': sv['0']['ec'],
                          'GPU': sv['0']['GPU'],
                          'allowed': (sv['0']['simulations'] - sum(sv['0']['status'])),
                          'forbidden': sum(sv['0']['status'])
                         }
        sv.close()
        #for i in range(0, self._simData0['simulations']):
        #    if self._simData0['Ekin'][2][i] < 90:
        #        self._simData0['Ekin'][0] = np.delete(self._simData0['Ekin'][0],i)
        #        self._simData0['Ekin'][1] = np.delete(self._simData0['Ekin'][1],i)
        #        self._simData0['Ekin'][2] = np.delete(self._simData0['Ekin'][2],i)
        print('Unshelve took '+str(time()-shelveStart)+' sec.')
        
        filterStart = time()
        mask = [self._simData0['Ekin'][1][i] > 0 for i in xrange(self._simData0['simulations'])]
        #[d[i] for i in xrange(len(d)) if c[i]]
        
        class MaskableList(list):
            def __getitem__(self, index):
                try: return super(MaskableList, self).__getitem__(index)
                except TypeError: return MaskableList(compress(self, index))
        mList = MaskableList
        
        sim2s = 0
        for m in mask:
            if m:
                sim2s += 1
        print(str(sim2s)+' out of '+str(self._simData0['simulations'])+' simulations made it through the filter.')
        
        if sim2s > 0:
            self._simData2 = {'simName': self._simData0['simName'],
                              'simulations': sim2s,
                              'fissionType': self._simData0['fissionType'],
                              'particles': self._simData0['particles'],
                              'coulombInteraction': self._simData0['coulombInteraction'],
                              'nuclearInteraction': self._simData0['nuclearInteraction'],
                              'ODEruns': self._simData0['ODEruns'],
                              'D0': self._simData0['D0'],
                              'ab': self._simData0['ab'],
                              'ec': self._simData0['ec'],
                              'GPU': self._simData0['GPU'],
                              'Q': self._simData0['Q'],
                              'allowed': self._simData0['allowed'],
                              'forbidden': self._simData0['forbidden'],
                              'D': mList(self._simData0['D'])[mask],
                              'r': np.array([mList(sublist)[mask] for sublist in self._simData0['r']]),
                              'v': np.array([mList(sublist)[mask] for sublist in self._simData0['v']]),
                              'r0': np.array([mList(sublist)[mask] for sublist in self._simData0['r0']]),
                              'v0': np.array([mList(sublist)[mask] for sublist in self._simData0['v0']]),
                              'TXE': self._simData0['TXE'],#np.array([mList(sublist)[mask] for sublist in self._simData0['TXE']]),
                              'Ec0': np.array([mList(sublist)[mask] for sublist in self._simData0['Ec0']]),
                              'Ekin0': np.array([mList(sublist)[mask] for sublist in self._simData0['Ekin0']]),
                              'angle': mList(self._simData0['angle'])[mask],
                              'Ec': np.array([mList(sublist)[mask] for sublist in self._simData0['Ec']]),
                              'Ekin': np.array([mList(sublist)[mask] for sublist in self._simData0['Ekin']]),
                              'status': mList(self._simData0['status'])[mask],
                              'error': mList(self._simData0['error'])[mask],
                              'wentThrough': mList(self._simData0['wentThrough'])[mask],
                              'Ekins': mList(self._simData0['Ekins'])[mask]
                              }
            print('Filter time took '+str(time()-filterStart))
            
            
            print([self._simData2['r0'][0][0],
                   self._simData2['r0'][1][0],
                   self._simData2['r0'][2][0],
                   self._simData2['r0'][3][0],
                   self._simData2['r0'][4][0],
                   self._simData2['r0'][5][0]])
            print([self._simData2['v0'][0][0],
                   self._simData2['v0'][1][0],
                   self._simData2['v0'][2][0],
                   self._simData2['v0'][3][0],
                   self._simData2['v0'][4][0],
                   self._simData2['v0'][5][0]])
        
        self._simData = self._simData0
        
        if self._verbose:
            print(str(self._simData['forbidden'])+" errors out of "+ \
                  str(self._simData['simulations'])+" simulations.")
            print("Ea max: "+str(np.max(self._simData['Ekin'][0])))
            print("Theta mean: "+str(np.mean(self._simData['angle'])))
            ccts = 0
            for i in range(0, self._simData['simulations']):
                if self._simData['angle'][i] < 2:
                    ccts += 1
            print(str(ccts)+' out of '+str(self._simData['simulations'])+' events were CCT (<2degrees).')
            print("%1.2f percent were CCT (<2degrees)." % (100.0*(float(ccts) / \
                                          float(self._simData['simulations']))))
            print('-----------------------------------------------------------')
            Etp_inf = np.mean(self._simData['Ekin'][0])
            Ehf_inf = np.mean(self._simData['Ekin'][1])
            Elf_inf = np.mean(self._simData['Ekin'][2])
            Eff_inf = np.mean(np.array(self._simData['Ekin'][1]) +\
                              np.array(self._simData['Ekin'][2]))
            Etp_sci = np.mean(self._simData['Ekin0'][0])
            Eff_sci = np.mean(np.array(self._simData['Ekin0'][1]) +\
                              np.array(self._simData['Ekin0'][2]))
            print("Quantity\tTheory\tSimulation\tAbs Dev\tRel Dev")
            print("Etp_inf \t15.7  \t%1.1f\t\t" % Etp_inf),
            print("%1.1f\t" % (abs(15.7 - Etp_inf))),
            print("%1.1f" % (abs(15.7 - Etp_inf)/15.7))
            print("Ehf_inf \t63.25 \t%1.2f\t\t" % Ehf_inf),
            print("%1.1f\t" % (abs(63.25 - Ehf_inf))),
            print("%1.1f" % (abs(63.25 - Ehf_inf)/63.25))
            print("Elf_inf \t92.85 \t%1.2f\t\t" % Elf_inf),
            print("%1.1f\t" % (abs(92.85 - Elf_inf))),
            print("%1.1f" % (abs(92.85 - Elf_inf)/92.85))
            print("Eff_inf \t155.5 \t%1.1f\t\t" % Eff_inf),
            print("%1.1f\t" % (abs(155.5 - Eff_inf))),
            print("%1.1f" % (abs(155.5 - Eff_inf)/155.5))
            
            print("Etp_sci \t3.0  \t%1.1f\t\t" % Etp_sci),
            print("%1.1f\t" % (abs(3.0 - Etp_sci))),
            print("%1.1f" % (abs(3.0 - Etp_sci)/3.0))
            print("Eff_sci \t13.0 \t%1.1f\t\t" % Eff_sci),
            print("%1.1f\t" % (abs(13.0 - Eff_sci))),
            print("%1.1f" % (abs(13.0 - Eff_sci)/13.0))
            print("Angle_mean\t82.0 \t%1.1f" % np.mean(self._simData['angle']))
            #print(self._simData['Ekin'])
            #print(self._simData['Q'])
            
            vtpmax = 0
            for vi in range(0,self._simData['simulations']):
                this_v = np.sqrt(self._simData['v'][0][vi]**2 + self._simData['v'][1][vi]**2)
                if this_v > vtpmax:
                    vtpmax = this_v
            print("v_tp max: "+str(vtpmax))
            print("E_lf max: "+str(max(self._simData['Ekin'][2])))
            print("LF: "+ self._simData['particles'][2].name)
            
    def plotItAll(self):
        """
        """
        r0tprel = np.zeros(self._simData['simulations'])
        for ri in range(0,self._simData['simulations']):
            r0tprel[ri] = (abs(self._simData['r0'][2][ri])-(self._simData['ab'][0]+self._simData['ab'][1]))/((self._simData['r0'][4][ri]-self._simData['r0'][2][ri])-(self._simData['ab'][0]+self._simData['ab'][2]))
        
        figNum = 0
        figNum += 1
        _plotEnergyDist(self._simData['simulations'],
                        np.array(self._simData['Ekin'][1]) + \
                        np.array(self._simData['Ekin'][2]),
                        self._simData['Ekin'][0],
                        self._simData['Q'],
                        figNum,
                        nbins=200)
        figNum += 1
        _plotProjectedEnergyDist(self._simData['Ekin'][0],figNum,'Energy distribution of ternary particle',nbins=100)
        """
        figNum += 1
        _plotProjectedEnergyDist(self._simData['Ekin'][1],figNum,'Energy distribution of heavy fragment',nbins=100)
        figNum += 1
        _plotProjectedEnergyDist(self._simData['Ekin'][2],figNum,'Energy distribution of light fragment',nbins=100)
        """
        figNum += 1
        _plotAngularDist(self._simData['angle'],figNum,nbins=100)
        figNum += 1
        _plotxyHist(r0tprel,self._simData['r0'][1],figNum,nbins=200)
        #_plotxyHist(-self._simData['r0'][2],self._simData['r0'][1],figNum,nbins=200)
        """
        figNum += 1
        _plot2DHist(x_in=self._simData['v0'][0],
                    y_in=self._simData['v0'][1],
                    figNum_in=figNum,
                    title_in='TP v0 distribution',
                    xlabel_in='vx [c]',
                    ylabel_in='vy [c]',
                    nbins=200)
        figNum += 1
        _plotDDistribution(self._simData['D'],figNum,nbins=100)
        """
        figNum += 1
        _plotEnergyAngleCorr(self._simData['angle'],self._simData['Ekin'][0],figNum,nbins=200)
        figNum += 1
        """
        _plot2DHist(x_in=self._simData['r0'][1],
                    y_in=self._simData['angle'],
                    figNum_in=figNum,
                    title_in='y vs angle',
                    xlabel_in='y [fm]',
                    ylabel_in='Angle [degrees]',
                    nbins=200)
        """
        plt.show()

    def plotCCT(self):
        figNum = 0
        """
        figNum += 1
        _plotAngularDist(self._simData['angle'],figNum,nbins=100)
        figNum += 1
        _plotEnergyAngleCorr(self._simData['angle'],self._simData['Ekin'][2],figNum,nbins=200)
        figNum += 1
        _plot2DHist(x_in=self._simData['angle'],y_in=self._simData['r0'][1],figNum_in=figNum,
                    title_in='Fission Axis offset versus Angle correlation',
                    xlabel_in='Angle [degrees]',
                    ylabel_in='Fission axis distance [fm]',
                    nbins=200)
        figNum += 1
        _plot2DHist(x_in=self._simData['r0'][1],y_in=self._simData['angle'],figNum_in=figNum,
                    title_in='Fission axis offset versus Angle correlation',
                    xlabel_in='Fission axis distance [fm]',
                    ylabel_in='Angle [degrees]',
                    nbins=200)
        """
        """
        figNum += 1
        _plot2DHist(x_in=-self._simData['r0'][2],y_in=self._simData['angle'],figNum_in=figNum,
                    title_in='x vs angle',
                    xlabel_in='x [fm]',
                    ylabel_in='Angle [degrees]',
                    nbins=200)
        figNum += 1
        _plot2DHist(x_in=-self._simData['r0'][2],y_in=self._simData['r0'][1],figNum_in=figNum,
                    title_in='x vs y',
                    xlabel_in='x [fm]',
                    ylabel_in='y [fm]',
                    nbins=200)
        figNum += 1
        _plot2DHist(x_in=(self._simData['r0'][4]-self._simData['r0'][2]),y_in=self._simData['angle'],figNum_in=figNum,
                    title_in='D vs angle',
                    xlabel_in='D [fm]',
                    ylabel_in='Angle [degrees]',
                    nbins=200)
        
        figNum += 1
        r0tprel = np.zeros(self._simData['simulations'])
        for ri in range(0,self._simData['simulations']):
            r0tprel[ri] = (abs(self._simData['r0'][2][ri])-(self._simData['ab'][0]+self._simData['ab'][1]))/((self._simData['r0'][4][ri]-self._simData['r0'][2][ri])-(self._simData['ab'][0]+self._simData['ab'][2]))
        _plot2DHist(x_in=(r0tprel),y_in=(self._simData['r0'][4]-self._simData['r0'][2]),figNum_in=figNum,
                    title_in='D vs relative TP displacement',
                    xlabel_in='x [percent]',
                    ylabel_in='D [fm]',
                    nbins=200)
        
        TXE0s = np.zeros(self._simData['simulations'])
        TXEs = np.zeros(self._simData['simulations'])
        allen = np.zeros(self._simData['simulations'])
        ecleft = np.zeros(self._simData['simulations'])
        for ri in range(0,self._simData['simulations']):
            TXE0s[ri] = self._simData['Q']-(self._simData['Ekin0'][0][ri]+ \
                                            self._simData['Ekin0'][1][ri]+ \
                                            self._simData['Ekin0'][2][ri]) - \
                                           (self._simData['Ec0'][0][ri] + \
                                            self._simData['Ec0'][1][ri] + \
                                            self._simData['Ec0'][2][ri])
            TXEs[ri] = self._simData['Q']-(self._simData['Ekin'][0][ri]+ \
                                           self._simData['Ekin'][1][ri]+ \
                                           self._simData['Ekin'][2][ri]) - \
                                          (self._simData['Ec'][0][ri] + \
                                           self._simData['Ec'][1][ri] + \
                                           self._simData['Ec'][2][ri])
            allen[ri] = self._simData['Ec'][0][ri] + self._simData['Ec'][1][ri] + self._simData['Ec'][2][ri] + \
                    self._simData['Ekin'][0][ri] + self._simData['Ekin'][1][ri] + self._simData['Ekin'][2][ri]
            ecleft[ri] = self._simData['Ec'][0][ri] + self._simData['Ec'][1][ri] + self._simData['Ec'][2][ri]
        figNum += 1
        _plot2DHist(x_in=(self._simData['Ekin'][2]),
                    y_in=(TXE0s),
                    figNum_in=figNum,
                    title_in='Ekin_LF_inf vs TXE0',
                    xlabel_in='Ekin_LF_inf [MeV]',
                    ylabel_in='TXE0 [MeV]',
                    nbins=200)
        figNum += 1
        _plot2DHist(x_in=(self._simData['Ekin'][2]),
                    y_in=(TXEs),
                    figNum_in=figNum,
                    title_in='Ekin_LF_inf vs TXE_static',
                    xlabel_in='Ekin_LF_inf [MeV]',
                    ylabel_in='TXE_static [MeV]',
                    nbins=200)
        """
        figNum += 1
        elf, ehf, eh = _plot2DHist(x_in=(self._simData['Ekin'][2]),
                    y_in=(self._simData['Ekin'][1]),
                    figNum_in=figNum,
                    title_in='Ekin_LF_inf vs Ekin_HF_inf',
                    xlabel_in='Ekin_LF_inf [MeV]',
                    ylabel_in='Ekin_HF_inf [MeV]',
                    nbins=200)
        figNum += 1
        _plot2DHist(x_in=(self._simData['Ekin'][2]),
                    y_in=(self._simData['Ekin'][0]),
                    figNum_in=figNum,
                    title_in='Ekin_LF_inf vs Ekin_TP_inf',
                    xlabel_in='Ekin_LF_inf [MeV]',
                    ylabel_in='Ekin_TP_inf [MeV]',
                    nbins=200)
        figNum += 1
        _plot2DHist(x_in=(self._simData['Ekin'][2]),
                    y_in=(self._simData['r0'][4]-self._simData['r0'][2]),
                    figNum_in=figNum,
                    title_in='Ekin_LF_inf vs D',
                    xlabel_in='Ekin_LF_inf [MeV]',
                    ylabel_in='D [fm]',
                    nbins=200)
        plt.figure(300)
        plt.plot(elf,ehf,'r--')
        plt.figure(303)
        plt.plot(eh,'r--')
        #plt.scatter(min(elf),max(ehf),marker='o',s='40',alpha=0.5)
        #plt.scatter(max(elf),min(ehf),marker='o',s='40',alpha=0.5)
        """
        figNum += 1
        _plot1DHist(x_in=ecleft,figNum_in=figNum,title_in='EC left',xlabel_in='EC [MeV]',nbins=100)
        figNum += 1
        _plot1DHist(x_in=allen,figNum_in=figNum,title_in='TKE + EC',xlabel_in='TKE + EC [MeV]',nbins=100)
        figNum += 1
        _plotProjectedEnergyDist(self._simData['Ekin'][2],figNum,'Light fragment.',nbins=100)
        figNum += 1
        _plotProjectedEnergyDist(self._simData['Ekin'][1],figNum,'Heavy fragment.',nbins=100)
        figNum += 1
        _plotProjectedEnergyDist(self._simData['Ekin'][0],figNum,'Ternary particle.',nbins=100)
        """
        p0_angles = []
        p0_y_angles = []
        p0_ys = []
        p0_ekins = []
        ps = 0
        for pi in range(0,self._simData['simulations']):
            #print(str(self._simData['v0'][0][pi])+'\t'+str(self._simData['v0'][1][pi])+'\t'+str(getAngle([self._simData['v0'][0][pi], self._simData['v0'][1][pi]],[1,0])))
            if not np.allclose(0, self._simData['v0'][0][pi]) and not np.allclose(0, self._simData['v0'][1][pi]):
                p0_angles.append(getAngle([self._simData['v0'][0][pi], self._simData['v0'][1][pi]],[1.0,0.0]))
                p0_y_angles.append(self._simData['angle'][pi])
                p0_ys.append(self._simData['r0'][1][pi])
                p0_ekins.append(self._simData['Ekin0'][0][pi])
                ps += 1
        
        print ps
        if ps > 0:
            figNum += 1
            _plot2DHist(x_in=p0_angles,y_in=p0_y_angles,figNum_in=figNum,
                        title_in='p0_tp_angle vs angle',
                        xlabel_in='p0 angle [degrees]',
                        ylabel_in='Angle [degrees]',
                        nbins=200)
            figNum += 1
            _plot2DHist(x_in=p0_angles,y_in=p0_ys,figNum_in=figNum,
                        title_in='p0_tp_angle vs y',
                        xlabel_in='p0 angle [degrees]',
                        ylabel_in='y [fm]',
                        nbins=200)
            figNum += 1
            _plot2DHist(x_in=p0_angles,y_in=p0_ekins,figNum_in=figNum,
                        title_in='p0_tp_angle vs Ekin_0_tp',
                        xlabel_in='p0 angle [degrees]',
                        ylabel_in='Ekin_0_tp [MeV]',
                        nbins=200)
        #print("Q: "+str(self._simData['Q']))
        #print("Ec0: "+str(self._simData['Ec0']))
        #print("LF: "+str(self._simData['Ekin'][2]))
        #print("HF: "+str(self._simData['Ekin'][1]))
        #print("TP: "+str(self._simData['Ekin'][0]))
    def plotTrajectories(self, color):
        """
        """
        figNum = 0
        figNum += 1

        # Check that file exists
        if self._simulationPath == "" or not os.path.isfile(self._simulationPath + "shelvedTrajectories.sb"):
            raise ValueError("simulationPath must be an existing and valid .sb "
                             "file.")
        # Unshelve
        print('Starting unshelve.')
        shelveStart = time()
        sv = shelve.open(self._simulationPath + "shelvedTrajectories.sb")
        self._trajectoryData = {'simName': sv['0']['simName'],
                                'simulations': sv['0']['simulations'],
                                'trajectories': sv['0']['trajectories'],
                                'odeSteps': sv['0']['odeSteps'],
                                'nbrOfParticles': sv['0']['nbrOfParticles']
                               }
        sv.close()

        print(np.shape(self._trajectoryData['trajectories']))        
        for i in range(0,self._trajectoryData['simulations']):
            for p in range(0,self._trajectoryData['nbrOfParticles']):
                plt.plot(self._trajectoryData['trajectories'][i][2*p],
                         self._trajectoryData['trajectories'][i][2*p+1], color=color)
        #plt.show()
    
    def animateTrajectories(self):
        """
        """
        
        for i in range(0,self._trajectoryData['simulations']):
            r = self._trajectoryData['trajectories'][i]
            maxrad = max(self._trajectoryData['ab'])
            plt.axis([np.floor(np.amin([r[0],r[2],r[4]]))-maxrad,
                      np.ceil(np.amax([r[0],r[2],r[4]]))+maxrad,
                      min(np.floor(np.amin([r[1],r[3],r[5]])),-maxrad)-maxrad,
                      max(np.amax([r[1],r[3],r[5]]),maxrad)+maxrad])
            
            skipsize = 5000
            for i in range(0,int(len(r[0])/skipsize)):
                plt.clf()
                plt.axis([np.floor(np.amin([r[0,],r[2],r[4]]))-maxrad,
                          np.ceil(np.amax([r[0],r[2],r[4]]))+maxrad,
                          min(np.floor(np.amin([r[1],r[3],r[5]])),-maxrad)-maxrad,
                          max(np.amax([r[1],r[3],r[5]]),maxrad)+maxrad])
                plotEllipse(r[0][i*skipsize],r[1][i*skipsize],self._sa.ab[0],self._sa.ab[1])
                plotEllipse(r[2][i*skipsize],r[3][i*skipsize],self._sa.ab[2],self._sa.ab[3])
                plotEllipse(r[4][i*skipsize],r[5][i*skipsize],self._sa.ab[4],self._sa.ab[5])
                plt.plot(r[0][0:i*skipsize],r[1][0:i*skipsize],'r-',lw=2.0)
                plt.plot(r[2][0:i*skipsize],r[3][0:i*skipsize],'g:',lw=4.0)
                plt.plot(r[4][0:i*skipsize],r[5][0:i*skipsize],'b--',lw=2.0)
                
                plt.draw()
            plt.show()
    
    def plotInitialConfigurations(self):
        """
        """
        # Check that file exists
        if self._simulationPath == "" or not os.path.isfile(self._simulationPath + "initialConfigs.sb"):
            raise ValueError("simulationPath must be an existing and valid .sb "
                             "file.")
################################################################################
#                                  Ea vs Ef                                    #
################################################################################
def _plotEnergyDist(sims_in,Ef_in,Ea_in,Q_in,figNum_in,nbins=10):
    H, xedges, yedges = np.histogram2d(Ef_in,Ea_in,bins=nbins)
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
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
            maxIndex = sims_in
        if i < ymin:
            ymin = i
            minIndex = sims_in

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
    print("Angle_cent\t82.0 \t%1.1f\t\t" % bincenters[maxIndex]),
    print("%1.1f\t" % (abs(82.0 - bincenters[maxIndex]))),
    print("%1.1f" % (abs(82.0 - bincenters[maxIndex])/82.0))

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
#                                2D PIXEL SCATTER                              #
################################################################################
def _plot2DpixelScatter(xa_in,ya_in,xf_in,yf_in,z_in,figNum_in,title_in,xlabel_in,ylabel_in,label_in,):
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
    ax.set_title(title_in)
    ax.set_xlabel(xlabel_in)
    ax.set_ylabel(ylabel_in)
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
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
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
#                             plot a 1d histogram                              #
################################################################################
def _plot1DHist(x_in,figNum_in,title_in,xlabel_in,nbins=100):
    fig = plt.figure(figNum_in)
    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(x_in, bins=nbins)
    bincenters = 0.5*(bins[1:]+bins[:-1])
    # add a 'best fit' line for the normal PDF
    #y = mlab.normpdf( bincenters)
    l = ax.plot(bincenters, n, 'r--', linewidth=1)
    ax.set_title(title_in)
    ax.set_xlabel(xlabel_in)
    ax.set_ylabel('Counts')
################################################################################
#                             plot a 2d histogram                              #
################################################################################
def _plot2DHist(x_in,y_in,figNum_in,title_in,xlabel_in,ylabel_in,nbins=100):
    H, xedges, yedges = np.histogram2d(x_in,y_in,bins=nbins)
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
    fig = plt.figure(figNum_in)
    ax = fig.add_subplot(111)
    plt.pcolormesh(xedges,yedges,Hmasked)
    plt.title(title_in)
    plt.xlabel(xlabel_in)
    plt.ylabel(ylabel_in)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    return xedges, yedges, H
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
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
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

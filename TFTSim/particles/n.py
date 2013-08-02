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

class N:
    """
    Contains basic properties of a neutron, mostly used as a projectile
    particle inducing fission.
    """
    def __init__(self):
        self.name = "n"
        self.fullName = "Neutron"
        self.mEx = 8.071 # MeV/c^2 Mass excess
        self.Z = 0 # Proton number
        self.A = 1 # Atomic Mass number


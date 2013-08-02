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

class Cf252:
    """
    Contains basic properties of a Californium-252 (Cf252) particle, usually
    used as a fission target.
    """
    def __init__(self):
        self.name = "Cf252"
        self.fullName = "Californium2521"
        self.mEx = 76.035 # MeV/c^2 Mass excess
        self.Z = 98 # Proton number
        self.A = 252 # Atomic Mass number


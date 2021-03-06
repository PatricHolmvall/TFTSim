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

class Cd118:
    """
    Contains basic properties of a Cadmium-118 (Cd118) particle.
    """
    def __init__(self):
        self.name = "Cd118"
        self.fullName = "Cadmium118"
        self.mEx = -86.702 # MeV/c^2 mass excess
        self.Z = 48 # Proton number
        self.A = 118 # Atomic Mass number


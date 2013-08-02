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

class Sn132:
    """
    Contains basic properties of a Tin-132 (Sn132) particle.
    """
    def __init__(self):
        self.name = "Sn132"
        self.fullName = "Tin132"
        self.mEx = -76.5439 # MeV/c^2 Mass excess
        self.Z = 50 # Proton number
        self.A = 132 # Atomic Mass number


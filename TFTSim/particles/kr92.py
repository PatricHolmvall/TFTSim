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

class Kr92:
    """
    Contains basic properties of a Krypton-92 (Kr92) particle.
    """
    def __init__(self):
        self.name = "Kr92"
        self.fullName = "Krypton92"
        self.mEx = -68.7693 # MeV/c^2 mass excess
        self.Z = 36 # Proton number
        self.A = 92 # Atomic Mass number


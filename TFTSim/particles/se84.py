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

class Se84:
    """
    Contains basic properties of a Selenium-84 (Se84) particle.
    """
    def __init__(self):
        self.name = "Se84"
        self.fullName = "Selenium84"
        self.mEx = -75.9477 # MeV/c^2 mass excess
        self.Z = 34 # Proton number
        self.A = 84 # Atomic Mass number


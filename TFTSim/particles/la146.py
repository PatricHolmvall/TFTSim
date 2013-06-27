# This file is part of TFTSim.
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

class La146:
    """
    Contains basic properties of a Lanthanum-146 (La146) particle.
    """
    def __init__(self):
        self.name = "La146"
        self.fullName = "Lanthanum146"
        self.mEx = -69.050 # MeV/c^2 mass excess
        self.Z = 57 # Proton number
        self.A = 146 # Atomic Mass number


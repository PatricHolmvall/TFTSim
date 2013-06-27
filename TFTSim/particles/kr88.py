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

class Kr88:
    """
    Contains basic properties of a Krypton-88 (Kr88) particle.
    """
    def __init__(self):
        self.name = "Kr88"
        self.fullName = "Krypton88"
        self.mEx = -79.6913 # MeV/c^2 mass excess
        self.Z = 36 # Proton number
        self.A = 88 # Atomic Mass number


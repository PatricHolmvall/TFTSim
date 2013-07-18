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

class Ni72:
    """
    Contains basic properties of a Nickel-72 (Ni72) particle.
    """
    def __init__(self):
        self.name = "Ni72"
        self.fullName = "Nickel72"
        self.mEx = -54.2261 # MeV/c^2 mass excess
        self.Z = 28 # Proton number
        self.A = 72 # Atomic Mass number


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

class H3:
    """
    Contains basic properties of a Hydrogen-3 nucleus (triton).
    """
    def __init__(self):
        self.name = "H3"
        self.fullName = "Hydrogen3"
        self.mEx = 14.9498061 # MeV Mass excess
        self.Z = 1 # Proton number
        self.A = 3 # Atomic Mass number

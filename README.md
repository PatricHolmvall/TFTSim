[![DOI](https://zenodo.org/badge/10411969.svg)](https://zenodo.org/badge/latestdoi/10411969)

TFTSim
========

TFTSim is a tool to simulate (classically) trajectories of fragments produced
in a Ternary Fission process, based on their Coulomb interaction right after
fissioning. TFTSim can be used to simulate a wide range of basic starting
configurations with varying geometrical and intial kinetic energy, based on
energy and total momentum conservation. The program uses a simple ode solver to
solve the equations of motion to produce particle trajectories, which are fit to
experimentally confirmed angular and energy distributions, in order to extract
information about what scission configurations are realistic.

TFTSim was last developed on Sep 13, 2013

----


System Requirements
----
+ Python 2.6 with [numpy](http://numpy.scipy.org/),
  [SciPy](http://www.scipy.org/) and
  [SimPy](http://simpy.sourceforge.net/).

Downloading
----

Source files are available at https://github.com/PatricHolmvall/TFTSim


Authors
----

+ Patric Holmvall - patric.hol@gmail.com

Citing
----

If you have used TFTSim, or in general find it useful, please consider citing it: [DOI: 10.5281/zenodo.4818457](https://zenodo.org/badge/latestdoi/10411969)

Licensing
----

TFTSim is free software.  See the file COPYING for copying conditions

-------------------------------------------------------------------------------
This file is part of TFTSim.

TFTSim is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

TFTSim is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with TFTSim.  If not, see <http://www.gnu.org/licenses/>.

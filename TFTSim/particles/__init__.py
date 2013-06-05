"""
The particle system package contains classes that represent different particles
and their basic properties (e.g. mass, charge).

Currently a particle should have an init that defines the following variables:
1) self.name: Short abbreviated nuclear name (specifying particle and A)
2) self.fullName: Full human readable name, typically used in legends of plots
3) self.mEx: Excess mass of the particle, remember to keep fissioning particles
             and fission fragments appart as they should have different signs.
4) self.Z: Proton number
5) self.A: Atomic Mass number
"""


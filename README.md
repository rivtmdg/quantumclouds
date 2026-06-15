# quantumclouds

# QuantumClouds
QuantumClouds is a lightweight, efficient Python library designed for computing and visualizing atomic orbital electron density distributions.

# Installation
Install the latest version directly from PyPI:
pip install quantumclouds

# Quick Start
Here is how to get up and running with QuantumClouds:


from quantumclouds.engine import AtomicOrbital

carbon = AtomicOrbital("C", l=3, m=1, grid_size=100)

carbon.compute()

carbon.plot3d(cmap='magma')

# Features
Physics Engine: Calculates electron density for hydrogen-like atoms.

Constants: Built-in table for effective nuclear charges (Z_eff).

Visualization: Built-in tools for 3D visualization.

License
Distributed under the MIT License.

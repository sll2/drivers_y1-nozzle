# CEESD-Y1_nozzle

Simulation of compressible fluid flow through a converging-diverging nozzle using MIRGE-Com.

The main driver is nozzly.py with the problem setup in [baseline](baseline) being the generally accepted way to run the simulation.
  
Simulation data (i.e. meshes) are located in [data](data)

Numerical experiments and/or driver variations can be located in [experiments](experiments), these are variations that may or may not derive from the current driver in baseline, although they generally have common ancestery.
  
The driver/data used to create the timing data is located in [timing_runs](timing_runs), and is a smaller version of the full baseline run.

Associated documentation can be found [here](https://docs.google.com/document/d/1wL_1nLRqoCRpz9jtx-HYr5DrKDmZneOf-G4bKLg5IPk/edit?usp=sharing). This document outlines the specifics of the problem geometry and setup, as well as providing sample results and investigations using the driver.

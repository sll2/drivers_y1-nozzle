#!/bin/bash
mpirun -n 1 python -u -O -m mpi4py nozzle.py -i restart_params.yaml -r restart_data/nozzle-000010

#!/bin/bash
mpirun -n 1 python -u -m mpi4py ./nozzle.py -r nozzle-100010-0000.pkl -i run_params.yaml

#!/bin/bash
mpirun -n 1 python -u -m mpi4py ./nozzle.py -i run_params.yaml

#!/bin/bash


if [ -z "$(ls -A emirge)" ]; then
  git clone git@github.com:illinois-ceesd/emirge.git emirge
  #echo "no emirge"
else
  echo "emirge install already present. Remove to build anew"
fi

cd emirge

if [ -z ${CONDA_PATH+x} ]; then
  echo "CONDA_PATH unset, installing new conda with emirge"
  ./install.sh --env-name=mirgeDriver.Y1nozzle --conda-env=../myenv.yml --pip-pkgs=../myreqs.txt
else
  echo "Using existing Conda installation, ${CONDA_PATH}"
  ./install.sh --conda-prefix=$CONDA_PATH --env-name=mirgeDriver.Y1nozzle --conda-env=../myenv.yml --pip-pkgs=../myreqs.txt
fi


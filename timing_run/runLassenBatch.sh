#! /bin/bash --login
#BSUB -nnodes 1
#BSUB -G uiuc
#BSUB -W 720
#BSUB -J mirge_Y0
#BSUB -q pbatch
#BSUB -e runOutput.txt
#BSUB -o runOutput.txt

module load gcc/7.3.1
module load spectrum-mpi
conda deactivate
conda activate mirgeDriver.Y1nozzle
export PYOPENCL_CTX="port:tesla"
#export PYOPENCL_CTX="0:2"
jsrun_cmd="jsrun -g 1 -a 1 -n 1"
export XDG_CACHE_HOME="/tmp/$USER/xdg-scratch"
$jsrun_cmd js_task_info
# file for stdout
outputFile=`$HOME/bin/getProgOutName.sh mirge`
echo "Writing output to ${outputFile}"

$jsrun_cmd python -u -O -m mpi4py ./nozzle.py -i timing_run_params.yaml > ${outputFile}

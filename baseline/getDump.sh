#! /bin/bash
#
#  This script finds the highest-numbered Mirge output file
#
fileList=(`ls -a restart_data/nozzle*.pkl`)
dumpFile=${fileList[${#fileList[@]}-1]}
dumpIndex=(`echo $dumpFile | cut -f2 -d -`)
#echo "Most recent dump index is $dumpIndex"
echo $dumpFile
echo $dumpIndex

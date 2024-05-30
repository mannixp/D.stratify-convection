#!/bin/bash
#
#
conda activate dedalus3

# To launch
# nohup ./Simulate.sh > HC.txt & 
# Ra # Good to pass this + configuration too

JOB_NAME="HC_Ra1e10_T2e04"
BASEPATH="/home/pmannix/Dstratify/DNS_RBC/"

rm -rf $JOB_NAME
mkdir -p $JOB_NAME"/"
cd "./"$JOB_NAME

# Files
SUBFOLD1=$BASEPATH$"/rayleigh_benard.py";
SUBFOLD2=$BASEPATH$"/plot_snapshots.py";

# Run simulations
mpiexec -np 8 python3 $SUBFOLD1

# Plot snapshots
mpiexec -np 1 python3 $SUBFOLD2  ./snapshots/*.h5
mkdir -p $"./Plotted_Data"

mv -v *.png $"./Plotted_Data" 
mv -v frames/ $"./Plotted_Data" 


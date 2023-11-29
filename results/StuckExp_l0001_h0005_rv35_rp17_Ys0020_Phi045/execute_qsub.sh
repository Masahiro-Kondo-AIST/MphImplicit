#!/bin/sh
#PBS -N StuckExp
#PBS -j oe
#PBS -l select=1:ncpus=1:ngpus=1:mem=128gb
#PBS -l walltime=960:00:00
 cd $PBS_O_WORKDIR
# export OMP_NUM_THREADS=12


#../../source/Mph_gcc bubble.data bubble.grid bubble%03d.prof bubble%03d.vtk bubble_acc.log 16
./execute_acc.sh
 
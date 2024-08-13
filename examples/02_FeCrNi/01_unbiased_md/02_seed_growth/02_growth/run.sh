#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH -n 25
#SBATCH --output=log.log

mkdir -p data/snapshot 
mkdir -p data/restart
mkdir -p data/logs
mkdir -p data/thermo

source /etc/profile
module load mpi/openmpi-4.1.5
lammps='/home/gridsan/ksheriff/INSTALLS/lammps/src/lmp_mpi'

T=800
mpirun -n 25 $lammps -in in.lmp -var RANDOM ${RANDOM} -log data/logs/log_${T}K.log -var a 3.593649980395461 -var v 11.603047551618317 -var T ${T} -var potentials_dir /home/gridsan/ksheriff/INSTALLS/potentials 

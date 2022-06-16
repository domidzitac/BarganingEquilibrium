#!/bin/bash 
#SBATCH --nodes=3                        # requests 3 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=50:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=barganing-equilibrium
#SBATCH --output=barganing-equilibrium.out

module purge
module load python/intel/3.8.6

python ./code_equil.py

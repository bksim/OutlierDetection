#!/bin/bash
#SBATCH --ntasks 1                #Number of processes
#SBATCH --nodes 1                  #Number of nodes
#SBATCH -t 100                 #Runtime in minutes
#SBATCH -p general            #Partition to submit to

#SBATCH --mem-per-cpu=200     #Memory per cpu in MB (see also --mem)
#SBATCH -o rcb_I.out    #File to which standard out will be written
#SBATCH -e rcb_I.err    #File to which standard err will be written

# load modules
module load centos6/numpy-1.7.1_python-2.7.3
module load centos6/pandas-0.11.0_python-2.7.3

python calculate_features.py -p rcb/phot/I -o rcb_I.csv
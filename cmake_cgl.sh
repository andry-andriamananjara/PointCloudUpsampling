#!/bin/bash

#SBATCH --job-name=cmake_cgl
#SBATCH --account=project_2009916
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:3,nvme:2

export SING_IMAGE=/projappl/project_2009916/Env/folder_cmake/cmakecgcal.sif
export SING_FLAGS=--nv

cd evaluation_code_puganTF
apptainer_wrapper exec bash all_non_uniform.sh
apptainer_wrapper exec bash all_uniform.sh
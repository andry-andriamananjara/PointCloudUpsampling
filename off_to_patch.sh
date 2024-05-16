#!/bin/bash

#SBATCH --job-name=Off_to_patch_knn
#SBATCH --account=project_2009916
#SBATCH --partition=gpu
##SBATCH --time=2-00:00:00
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:3,nvme:2

export PATH="/projappl/project_2009916/Env/venv_3dpytorch/bin:$PATH"

cd MC_5k
srun python off_to_xyz.py --isTrain=test --datasetdir=Mydataset/PU1K/non_uniform/test/original_meshes
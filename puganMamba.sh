#!/bin/bash

#SBATCH --job-name=trainpu1kGenDisP3Dconv_default_non_uniform
#SBATCH --account=project_2009906
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:3,nvme:2

export PATH="/projappl/project_2009906/Env/venv_3dpytorch/bin:$PATH"

cd train
##srun python train.py --exp_name=trainpu1kGen_uniform --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k --gen_attention=mamba
##srun python train.py --exp_name=trainpu1kGen_non_uniform --gpu=1 --use_gan --batch_size=12          --dataname=pu1k --gen_attention=mamba

##srun python train.py --exp_name=trainpu1kDis_uniform --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k --dis_attention=mamba
##srun python train.py --exp_name=trainpu1kDis_non_uniform --gpu=1 --use_gan --batch_size=12       --dataname=pu1k --dis_attention=mamba

##srun python train.py --exp_name=trainpu1kGenDis_uniform --gpu=1 --use_gan --batch_size=12  --uniform --dataname=pu1k --gen_attention=mamba --dis_attention=mamba
##srun python train.py --exp_name=trainpu1kGenDis_non_uniform --gpu=1 --use_gan --batch_size=12           --dataname=pu1k --gen_attention=mamba --dis_attention=mamba

## Mamba default value
##srun python train.py --exp_name=trainpu1kGenDis_default_uniform --gpu=2 --use_gan --batch_size=12  --uniform --dataname=pu1k --gen_attention=mamba --dis_attention=mamba
##srun python train.py --exp_name=trainpu1kGenDis_default_non_uniform --gpu=2 --use_gan --batch_size=12           --dataname=pu1k --gen_attention=mamba --dis_attention=mamba

############################# Feature extraction
##srun python train.py --exp_name=trainpu1kGenP3Dconv_default_uniform --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k --gen_attention=mamba --feat_ext=P3DConv
##srun python train.py --exp_name=trainpu1kGenP3Dconv_default_non_uniform --gpu=1 --use_gan --batch_size=12          --dataname=pu1k --gen_attention=mamba --feat_ext=P3DConv

##srun python train.py --exp_name=trainpu1kDisP3Dconv_default_uniform --gpu=1 --use_gan --batch_size=12 --uniform --dataname=pu1k --dis_attention=mamba --feat_ext=P3DConv
##srun python train.py --exp_name=trainpu1kDisP3Dconv_default_non_uniform --gpu=1 --use_gan --batch_size=12       --dataname=pu1k --dis_attention=mamba --feat_ext=P3DConv

##srun python train.py --exp_name=trainpu1kGenDisP3Dconv_default_uniform --gpu=1 --use_gan --batch_size=12  --uniform --dataname=pu1k --gen_attention=mamba --dis_attention=mamba --feat_ext=P3DConv
srun python train.py --exp_name=trainpu1kGenDisP3Dconv_default_non_uniform --gpu=1 --use_gan --batch_size=12           --dataname=pu1k --gen_attention=mamba --dis_attention=mamba --feat_ext=P3DConv

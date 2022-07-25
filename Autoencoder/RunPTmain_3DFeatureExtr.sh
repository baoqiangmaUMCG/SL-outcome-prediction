#!/bin/bash

#SBATCH --job-name=radioth_fm_feature_pyramid_mse_adv_0
#SBATCH --mail-type=END
##SBATCH --mail-user=b.ma@rug.nl
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
##SBATCH --partition=gpushort
##SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=PTmain_3DFeatureExtr_fm_%j.log

# module load OpenCV/3.4.4-foss-2018a-Python-3.6.4
# module load Python/3.6.4-fosscuda-2018a
# module load Tkinter/3.6.4-fosscuda-2018a-Python-3.6.4
# module load h5py/2.7.1-fosscuda-2018a-Python-3.6.4
module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4

source /data/pg-dl_radioth/venv_pytorch/bin/activate
cd /data/pg-dl_radioth/scripts/Autoencoder_opcradiomics/model_clean/

python "/data/pg-dl_radioth/scripts/Autoencoder_opcradiomics/model_clean/PTmain_3DFeatureExtr.py" --input_type 2
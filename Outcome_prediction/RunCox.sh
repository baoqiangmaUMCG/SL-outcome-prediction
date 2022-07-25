#!/bin/bash

#SBATCH --job-name=CoxRegresion_bootstrapping
#SBATCH --mail-type=END
##SBATCH --mail-user=b.ma@rug.nl
#SBATCH --time=4:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --partition=gpushort
##SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --output=PTmain_3DFeatureExtr_fm_%j.log

# module load OpenCV/3.4.4-foss-2018a-Python-3.6.4
# module load Python/3.6.4-fosscuda-2018a
# module load Tkinter/3.6.4-fosscuda-2018a-Python-3.6.4
# module load h5py/2.7.1-fosscuda-2018a-Python-3.6.4
module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4

source /data/pg-dl_radioth/venv_pytorch/bin/activate
cd /data/pg-dl_radioth/scripts/Autoencoder_opcradiomics/model_clean/Outcome_prediction/
python "/data/pg-dl_radioth/scripts/Autoencoder_opcradiomics/model_clean/Outcome_prediction/forward_combined_feature_selection.py" --input_type 3 --outcome  0  --sub  1
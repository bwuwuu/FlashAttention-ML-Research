#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --output=gpujob.out
#SBATCH --gres=gpu:a100:1

module purge
module load gcc/11.3.0
module load cuda/11.6.2
module load cudnn/8.4.0.27-11.6
module load conda

eval "$(conda shell.bash hook)"
conda activate llama2

# Check if torch is installed
if ! python3 -c "import torch" &> /dev/null; then
    # install torch manually
    echo "PyTorch has not been installed. Install PyTorch first."
else
    echo "PyTorch is already installed."
fi

if ! python3 -c "import ninja" &> /dev/null; then
    pip3 install ninja
    echo "Ninja has been installed."
else
    echo "Ninja is already installed."
fi

python3 test.py
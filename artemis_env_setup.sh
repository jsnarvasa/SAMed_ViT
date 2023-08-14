#!/bin/bash
module load python/3.8.2
virtualenv --system-site-packages ~/venv_samed

source ~/venv_samed/bin/activate

pip install icecream==2.1.3
pip install einops==0.6.1
pip install safetensors==0.3.1

# IVC2627

# CONDA
## Create conda environment

```
conda create -n IVC2627P310Env python=3.10
```

or

```
conda create -p PATH python=3.10
```
## Activate conda environment
```
conda activate IVC2627P310Env
```
# Install packages
```
conda install jupyter

conda install pillow

conda install scikit-image

conda install matplotlib

conda install tk

pip install opencv-contrib-python

conda install tqdm numba
```

# For Deep Learning
```
pip install ultralytics
pip install filterpy
```

# Fix YOLO for GPU
```
pip uninstall torch torchvision torchaudio

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# For Tensorflow
```
conda create -n TF210 --file environment.yml

conda activate TF210

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

pip install "tensorflow<2.11"

pip install tensorflow_datasets
```
## Export conda environment
```
conda env export > environment.yml
```
## Import conda environment
```
conda env create -n Project_Environment_Name --file environment.yml
```
# PIP

## Import pip environment
```
pip install -r requirements.txt
```
## Export pip environment
```
pip freeze > requirements.txt
```
## Unofficial Windows Binaries for Python Extension Packages
<https://www.lfd.uci.edu/~gohlke/pythonlibs/>

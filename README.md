# CasDO
Paper: Information Cascade Prediction via Probabilistic Diffusion
### Requirements
The code was tested with `python 3.7`, `pyorch-gpu 1.10`, `cudatookkit 10.2`, and `cudnn 7.6.5`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name CasDO python=3.7 cudatoolkit=10.2 cudnn=7.6.5

# activate environment
conda activate CasDO

# install other requirements
pip install -r requirements.txt
```
### Run the code
```shell
cd ./CasDO

# generate information cascades
python gene_cas.py --input=./dataset/twitter/

# generate cascade graph and global graph embeddings 
python gene_emb.py --input=./dataset/twitter/

# run CasDO model
python run_model.py --input=./dataset/twitter/
```
More running options are described in the codes, e.g., `--input=./dataset/twitter/`

## Datasets

Datasets download link: [Google Drive](https://drive.google.com/file/d/1o4KAZs19fl4Qa5LUtdnmNy57gHa15AF-/view?usp=sharing) or [Baidu Drive (password: `1msd`)](https://pan.baidu.com/s/1tWcEefxoRHj002F0s9BCTQ).

## Folder Structure

```CasDO
└── utils: # The file includes each part of the module in CasDO
    ├── gene_cas.py # The core source code of building the cascade graph
    ├── gene_emb.py # The core source code of generating the tructural embedding of the cascade graph
    ├── run.py # Run model for training, validation, and test
    

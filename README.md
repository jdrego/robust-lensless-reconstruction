# Robust lensless image reconstruction via PSF estimation

Repository for our paper: 
  
_"Robust lensless image reconstruction via PSF estimation"_, Joshua D. Rego, Karthik Kulkarni, Suren Jayasuriya

Arizona State University

## Requirements:

The python package requirements are included in the file `requirements.yml`. We recommend installing the required packages in a separate virtual python environment using Anaconda: 

[Conda Download](https://www.anaconda.com/products/individual), 
[Conda Installation Instruction](https://docs.anaconda.com/anaconda/install/index.html)

After conda is correctly installed, run `conda activate` in the terminal, then make sure you are inside the project folder in the terminal and run the following command line which will create a new virtual environment named `lenslessGAN`: 

    conda env create -f requirements.yml

After all packages are installed you can switch into this virtual environment with:
    
    conda activate lenslessGAN

## Instructions:

Run the following in a command line at the root project folder:

    python run.py
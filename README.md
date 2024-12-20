## Description
Median housing value prediction

The housing data can be downloaded from <https://github.com/ageron/handson-ml/tree/master/datasets>. 
The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
First, install conda using Miniforge. You can follow these steps:
<https://github.com/conda-forge/miniforge?tab=readme-ov-file#install>

You can create a new conda environment using the `env.yml` file present in the root directory of this repo.
```SHELL
conda env create -f env.yml
```
The environment will be named as `mle-dev`.


Activate the environment
``` SHELL
conda activate mle-dev
```

Run the python file using the following command
``` SHELL
python3 nonstandardcode.py
```


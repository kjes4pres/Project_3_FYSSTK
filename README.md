# Project_3_FYSSTK
### Fall 2025
**Authors:** *Martine Jenssen Pedersen, Kjersti Stangeland, & Sverre Manu Johansen*

This is a collaboration repository for project 3 in FYS-STK4155. The aim of this project is to study a classification problem, specifically on determining weather types based on meteorological data, with neural networks and logistic regression.

### How to install required packages
A `requirements.txt` file is located in the repository. To reproduce our results, use the packages listed here. To install the packages, download the `requirements.txt` file, open your terminal and locate your project repository where you placed the downloaded file, in the command line write "´pip install -r requirements.txt´" or if you're using a conda environment type `conda install --file requirements.txt`.

### Overview of contents
The repository is organized as follows:

Functions and modules used for obtaining the results:
* `Code/functions/activation_funcs.py`: Activation functions and their derivatives for our own-built FFNN.
* `Code/functions/cost_functions.py`: Cross entropy cost function and its derivative for our own-built FFNN.
* `Code/functions/ffnn.py`: Our own-built FFNN class.
* `Code/functions/make_dataset.py`: Class for the PyTorch built NN for reading in the dataset in a PyTorch-compatible way.
* `Code/functions/nn_pytorch.py`: PyTorch based NN class.

Notebooks for running the code and plotting:
* `Code/main/LogisticRegression.ipynb`: Results for logistic regression using FFNN and PyTorch NN with one layer, on weather class prediction.
* `Code/main/NeuralNetworks.ipynb`: Results for weather type prediction using FFNN and PyTorch NN.
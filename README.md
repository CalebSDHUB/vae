# VAE Project

A Variational Autoencoder (VAE) implementation for generative modeling and representation learning.

## Overview

This project implements a Variational Autoencoder (VAE), a deep learning model that combines neural networks with variational inference to learn compressed representations of data.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/CalebSDHUB/vae
cd <project_directory>
```

## Requirements

This project uses Python 3.11.11 with `virtualenv` for package management.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Running the Demo Notebook

The project includes a demonstration notebook for training and evaluating the VAE model.

### Training the VAE

To train the VAE model from scratch:

The training script will:
- Load the MNIST dataset from `data_mnist/`
- Initialize the VAE model
- Train the model using the loss functions defined in `loss_criterion.py`
- Save checkpoints to the `ckpt/` directory


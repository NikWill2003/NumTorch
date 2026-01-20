NumTorch (WIP) — a tiny PyTorch-like framework in NumPy
=======================================================

NumTorch is an educational mini deep learning framework built from scratch using NumPy.
It includes a small autograd Tensor, a Module system, common layers, optimisers, losses,
a simple DataLoader, and an MNIST training example.

Features (current)
------------------
- Autograd Tensor with reverse-mode backprop over a dynamic compute graph (topo sort + _backward closures).
- Module API with nested module discovery + parameter collection (params).
  Includes Sequential, Affine, Relu, DropOut (WIP), SoftMax (temporary).
- Optimisers: SGD, Adam.
- Losses: SoftMaxCrossEntropy, CrossEntropy, MeanSquaredError.
- Training loop: Trainer with accuracy metric + optional Weights & Biases logging.
- MNIST utilities: auto-downloads mnist.npz, supports normalisation + flattening, plus a simple DataLoader.

Quickstart
----------
1) Install dependencies
- numpy
- wandb
- python-dotenv (only needed if you want to load WANDB_API_KEY from a .env file)

2) Run MNIST example
- The provided mnist.py script builds a small MLP and trains it using Trainer, Adam, and SoftMaxCrossEntropy.
- MNIST will be downloaded to datasets/mnist.npz on first run.


Then run:
python mnist.py

Project structure (approx.)
---------------------------
- tensor.py    — Tensor, Parameter, autograd ops
- modules.py   — Module, layers, Sequential
- optim.py     — optimisers + loss functions
- datasets.py  — DataLoader, MNIST download/load
- utils.py     — Trainer, metrics, W&B logging
- config.py    — dtype/seed config + RNG getter

Why this exists
---------------
This project is primarily for learning: implementing autograd, modules, and optimisation
from first principles to understand how modern deep learning frameworks work under the hood.

# Modeling Predator-Prey Interactions from Real Fish Videos: A Generative Adversarial Imitation Learning Approach

This thesis investigates whether predator-prey interactions, particularly pursuit and escape, can be recovered from real video recordings using imitation learning.

This repository contains the code implementing my thesis methodology.

Notebooks 1.1 - 1.3 cover data selection + labeling. 

Notebooks 2.1 - 2.4 cover creating expert tensors + pre-training encoder/policy with behavioral cloning.

Notebooks 3.1 - 3.3 cover model training.

Notebook 4.1 is the evaluation of the learned policy.

The models and utils folders contain the network architectures and the helper functions used across the notebooks.

The data folder contains both the raw and processed data.

The figures and videos folder contains the training results.

To run the notebooks, install the dependencies in requirements.txt.
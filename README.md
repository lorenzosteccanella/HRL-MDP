# HRL-MDP
This repository presents the code of the paper: 

Hierarchical Representation Learning for Markov Decision Processes  

Proceedings of The 2nd Conference on Lifelong Learning Agents, 2023 

https://proceedings.mlr.press/v232/steccanella23a.html

To run an example on colab click to: [Colab Example](https://colab.research.google.com/github/lorenzosteccanella/HRL-MDP/blob/main/Example_colab.ipynb)


# HRL-MDP: Hierarchical Representation Learning for Markov Decision Processes

This repository contains the code for the paper:

**Hierarchical Representation Learning for Markov Decision Processes**

_Proceedings of The 2nd Conference on Lifelong Learning Agents, 2023_

[https://proceedings.mlr.press/v232/steccanella23a.html](https://proceedings.mlr.press/v232/steccanella23a.html)

## Overview

This project explores a method for learning hierarchical representations in Markov Decision Processes (MDPs).  The approach aims to discover abstract states and transitions, enabling more efficient planning and learning in complex environments. The code implements a soft clustering technique to extract these hierarchical structures from trajectory data.

## Repository Structure

*   `model.py`: Defines the neural network architecture for soft clustering.
*   `soft_cluster.py`: Contains the main training loop and evaluation logic.
*   `utils.py`: Implements utility functions for data collection, plotting, and representation scoring.
*   `replay.py`: Defines the experience replay buffer.
*   `main.py`: Script to run the training process with configurable parameters.
*   `Example.ipynb`: A Jupyter Notebook demonstrating a basic usage of the codebase.
*   `Example_colab.ipynb`: A Colab version of Example.ipynb.
*   `requirements.txt`: Lists the required Python packages.

## Requirements

To run the code, you will need the following Python packages. You can install them using `pip`:

```bash
pip install -r requirements.txt
```
## Google Colab Example

For a quick demonstration, you can use the provided Google Colab notebook:

![alt text](https://colab.research.google.com/assets/colab-badge.svg)

This notebook provides a pre-configured environment with all the necessary dependencies and a simplified training setup.

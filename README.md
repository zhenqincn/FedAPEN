# README
This repository contains the source code of experiments introduced in the paper entitled with **FedAPEN: Personalized Cross-silo Federated Learning with Adaptability to Statistical Heterogeneity**, which is accepted by **ACM SIGKDD Conference on Knowledge Discovery and Data Mining (SIGKDD), 2023**.

**Source code will be publicly available soon.**

Note that in this repository, `shared model` means `shared model` described in this paper.

`FedAPEN`, `FedEN`, `FML-AE`, `FML-EE`, `APFL`, `FML(Sha)`, `FML(Pri)`, `FedAvg-FT`, `FedAvg` and `Individual` can be conducted by running `main.py` with different value of `algorithm` argument. 
+ `FedAPEN`: `python main.py --algorithm "learned_adaptive_training" ...`, where the records of accuracy can be revealed by `tools/plot_utils.py` with the name of `FedAPEN`.
+ `FedEN`: `python main.py --algorithm "equal_training" ...`, where the records of accuracy can be revealed by `tools/plot_utils.py` with the name of `FedAPEN`.
+ `FML-AE`: `python main.py --algorithm "learned_adaptive" ...`, where the records of accuracy can be revealed by `tools/plot_utils.py` with the name of `FML-AE`.
+ `FML-EE`: `python main.py --algorithm "learned_adaptive" ...`, where the records of accuracy can be revealed by `tools/plot_utils.py` with the name of `FML-EE`.
+ `APFL`: `python main.py --algorithm "APFL" ...`, where the records of accuracy can be revealed by `tools/plot_utils.py` with the name of `apfl`.
+ `FML(Sha)`: `python main.py --algorithm "learned_adaptive" ...`, where the records of accuracy can be revealed by `tools/plot_utils.py` with the name of `FML(Sha)`.
+ `FML(Pri)`: `python main.py --algorithm "learned_adaptive" ...`, where the records of accuracy can be revealed by `tools/plot_utils.py` with the name of `FML(Pri)`.
+ `FedAvg-FT`: `python main.py --algorithm "fed_avg_tune" ...`, where the records of accuracy can be revealed by `tools/plot_utils.py` with the name of `fed_avg_tune`.
+ `FedAvg`: `python main.py --algorithm "fed_avg" ...`, where the records of accuracy can be revealed by `tools/plot_utils.py` with the name of `fed_avg`.
+ `Individual`: `python main.py --algorithm "individual" ...`, where the records of accuracy can be revealed by `tools/plot_utils.py` with the name of `individual`.

`FedRep`, `FedAMP` and `FedProx` are implemented with an open source project (Temporarily hidden for review) provided in `PFL-Non-IID`.
They can be conducted by running `PFL-Non-IID/main.py` with different value of `algorithm` argument. 
To run `PFL-Non-IID/main.py`, you should first change directory from `root of this repository` to `PFL-Non-IID`.
Then, execute the commands listed below on your demand.
+ `FedRep`: `python main.py --algorithm "FedRep" ...`, where the records of accuracy can be revealed by `tools/plot_utils.py`.
+ `FedAMP`: `python main.py --algorithm "FedAMP" ...`, where the records of accuracy can be revealed by `tools/plot_utils.py`.
+ `FedProx`: `python main.py --algorithm "FedProx" ...`, where the records of accuracy can be revealed by `tools/plot_utils.py`.

The functionalities of directories and files contained in this repository are listed as below:
+ `models`: Defining the architectures of models.
+ `tools`: Containing some tools helping with this repository.
  + `nn_utils.py`: utilities for neural networks.
  + `plot_utils.py`: tools to calculate BMTA and plot the trend of mean accuracy .
  + `utils.py`: tools to decay learning rate.
+ `PFL-Non-IID`: This directory contains an open source repository for some personalized federated learning approaches. Note that `FedRep`, `FedAMP` and `FedProx` are implemented in this repository.
+ `data_loader.py`: loading dataset and partitioning them to these clients.
+ `main.py`: the entrance of this project.
+ `node.py`: implementation of functionalities for a node of FL, including the server node and client node.
+ `recorder.py`: recording the test accuracy of each client during training.

## Requirements
Please see `requirements.txt` for details.
# Loss-Customised Probabilistic Energy Time Series Forecasts Using Automated Hyperparameter Optimisation

This repository contains the Python implementation of the approach to create loss-customised probabilistic
forecasts using automated hyperparmeter optimisation. This approach is presented in the following paper:
> Kaleb Phipps, Stefan Meisenbacher, Benedikt Heidrich, Marian Turowski, Ralf Mikut, and Veit Hagenmeyer. 2023.
> Loss-Customised Probabilistic Energy Time Series Forecasts Using Automated Hyperparameter Optimisation. In The 14th
> ACM International Conference on Future Energy Systems (e-Energy ’23). ACM, pp. tbd. https://doi.org/10.1145/3575813.3595204

## Repository Structure

This repository is structured in a few key folders:

- `base_pipelines`: This folder contains the code used to create the pipelines that are then executed for each data set.
- `data`: This folder contains the data used for the analyses in our paper.
- `modules`: This folder contains multiple pyWATTS modules that are included in the pipelines.
- `pipelines`: This folder contains the pipelines which can be executed to recreate the results in our paper.


## Installation

Before the proposed approach can be applied using a [pyWATTS](https://github.com/KIT-IAI/pyWATTS) pipeline, you need to
prepare a Python environment and download energy time series (if you have no data available).

### 1. Setup Python Environment

Perform the following steps:

- Set up a virtual environment of Python 3.10 using e.g. venv (`python3.10 -m venv venv`) or Anaconda (`conda create -n env_name python=3.10`).
- Possibly install pip via `conda install pip`.
- Install tensorflow via `pip install tensorflow` or if using a mac `pip install tensorflow-macos`.
- Install the dependencies with `pip install -r requirements.txt`.
- Install ray via `pip install -U "ray[default]==2.0.1"` with venv or `conda install -c conda-forge "ray-default==2.0.1"` with Anaconda
- Install tensorflow-addons via `pip install tensorflow-addons`.

### 2. Download Data (optional)

We provide the open source data to replicate our price, mobility, and solar results in the folder __data__.
However, if you want to replicate our electricity results, you have to download the
[ElectricityLoadDiagrams20112014 Data Set](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)  
from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/) and save it as __elec.csv__ in the data
folder.


## Execution
If you are interested in running code, you should navigate to the appropriate pipeline in the `pipelines` folder and run
the respective pipeline from there.

If you are interested in applying our method to your own data, you will need to create a new pipeline. You can use the
existing pipelines in the `pipelines` folder as orientation for any pipeline you create.


## Funding

This project is supported by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI and by the
Helmholtz Association under the Program “Energy System Design”.

## License

This code is licensed under the [MIT License](LICENSE).

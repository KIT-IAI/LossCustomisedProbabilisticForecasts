# Loss-Customised Probabilistic Energy Time Series Forecasts Using Automated Hyperparameter Optimisation

This repository contains the Python implementation of the approach to approach to create loss-customised probabilistic
forecasts using automated hyperparmeter optimisation. This approach is presented in the following paper:
> Kaleb Phipps, Stefan Meisenbacher, Benedikt Heidrich, Marian Turowski, Ralf Mikut, and Veit Hagenmeyer. 2023.
> Loss-Customised Probabilistic En- ergy Time Series Forecasts Using Automated Hyperparameter Optimisation. In The 14th
> ACM International Conference on Future Energy Systems (e-Energy ’23), June 20–23, 2023, Orlando, FL, USA. ACM, New York,
> NY, USA, 16 pages. https://doi.org/10.1145/3575813.3595204

## Repository Structure

TODO

## Installation

Before the propsed approach can be applied using a [pyWATTS](https://github.com/KIT-IAI/pyWATTS) pipeline, you need to
prepare a Python environment and download energy time series (if you have no data available).

### 1. Setup Python Environment

Perform the following steps:
- Set up a virtual environment using e.g. venv (`python -m venv venv`) or Anaconda (`conda create -n env_name`).
- Install the dependencies with `pip install -r requirements.txt`.
- Install tensorflow via `pip install tensorflow`.
- Install tensorflow-addons via `pip install tensorflow-addons`.

### 2. Download Data (optional)

We provide the open source data to replicate our price, mobility, and solar results in the folder __data__.
However, if you want to replicate our electricity results you have to download the
[ElectricityLoadDiagrams20112014 Data Set](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)  
from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/) and save it as __elec.csv__ in the data folder.


## Funding

This project is supported by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI, by the
Helmholtz Association under the Program “Energy System Design”.

## License

This code is licensed under the [MIT License](LICENSE).

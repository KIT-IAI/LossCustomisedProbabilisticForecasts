import logging
import warnings

import numpy as np
import xarray as xr
from copy import deepcopy
from enum import IntEnum
from typing import Dict, Union, Callable
import torch
import ray
from ray import tune
from ray.tune.tuner import Tuner

from modules.generator_base import GeneratorBase

from scipy import stats

from pywatts.core.base import BaseEstimator, BaseTransformer
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray, _get_time_indexes
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.exceptions import WrongParameterException
from pywatts.core.pipeline import Pipeline
from pywatts.core.pipeline_step import PipelineStep
from pywatts.core.run_setting import RunSetting
from pywatts.modules.wrappers.base_wrapper import BaseWrapper


class ProbForecastCINN(BaseTransformer):


    def __init__(self, name: str, cinn: GeneratorBase,
                 quantiles: list = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99],
                 sample_size: int = 100, sampling_std: float = 0.1):
        super().__init__(name)
        self.cinn = cinn
        self.quantiles = quantiles
        self.sample_size = sample_size
        self.sampling_std = sampling_std

    def get_params(self) -> Dict[str, object]:
        return {
            "cinn": self.cinn,
            "quantiles": self.quantiles,
            "sample_size": self.sample_size,
            "sampling_std": self.sampling_std
        }

    def set_params(self, cinn=None, quantiles=None, sample_size=None, sampling_std=None):
        if cinn is not None:
            self.cinn = cinn
        if quantiles is not None:
            self.quantiles = quantiles
        if sample_size is not None:
            self.sample_size = sample_size
        if sampling_std is not None:
            self.sampling_std = sampling_std

    def transform(self, input_data: xr.DataArray, **kwargs) -> xr.DataArray:
        x = input_data.values.reshape((len(input_data), -1))
        conds = self.cinn._get_conditions(kwargs)

        quantiles = {}

        z = self.cinn.generator.forward(torch.Tensor(x), torch.Tensor(conds), rev=False)[0]

        noise = torch.Tensor(self.sample_size * len(x), input_data.shape[-1]).normal_(mean=1,
                                                                                      std=self.sampling_std) * z.repeat(self.sample_size, 1)

        samples = self.cinn.generator.forward(noise, torch.Tensor(conds).repeat(self.sample_size, 1), rev=True)[0].detach().numpy()

        samples = samples.reshape(self.sample_size, len(x), -1)
        for k in self.quantiles:
            quantiles[k] = stats.scoreatpercentile(samples, k, axis=0)

        arr = np.array(list(quantiles.values()))
        arr = arr.swapaxes(0, 1)
        arr = arr.swapaxes(2, 1)
        da = xr.DataArray(arr, dims=[_get_time_indexes(input_data)[0], "horizon", "quantiles"],
                          coords={"quantiles": list(quantiles.keys()),
                                  _get_time_indexes(input_data)[0]: input_data.indexes[
                                      _get_time_indexes(input_data)[0]]})

        return da


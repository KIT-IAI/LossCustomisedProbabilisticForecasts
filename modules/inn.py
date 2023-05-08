import pickle
from abc import ABC
from typing import Dict

import numpy as np
import torch
import xarray as xr
from pywatts.core.filemanager import FileManager
from pywatts.utils._xarray_time_series_utils import _get_time_indexes
from scipy import stats

from modules.generator_base import GeneratorBase
from modules.inn_base_functions import INN


class INNWrapperBase(GeneratorBase, ABC):

    def __init__(self, name: str = "INN", logprob=False, std=0.1, quantiles=[50, 1, 99, 5, 95, 15, 85, 25, 75, 10, 90, 20, 80, 30, 70, 40, 60], sample_size=100, **kwargs):
        super().__init__(name, **kwargs)
        self.logprob = logprob
        self.std = std
        self.quantiles = quantiles
        self.sample_size = sample_size

    def get_params(self) -> Dict[str, object]:

        return {
            "epochs": self.epoch,
            "horizon": self.horizon,
            "cond_features": self.cond_features,
        }

    def set_params(self, epochs=None, horizon=None, cond_features=None):
        if epochs is not None:
            self.epoch = epochs
        if horizon is not None:
            self.horizon = horizon
        if cond_features is not None:
            self.cond_features = cond_features

    def save(self, fm: FileManager) -> Dict:
        """
        Saves the modules and the state of the module and returns a dictionary containing the relevant information.

        :param fm: the filemanager which can be used by the module for saving information about the module.
        :type fm: FileManager
        :return: A dictionary containing the information needed for restoring the module
        :rtype:Dict
        """
        json_module = super().save(fm)
        path = fm.get_path(f"module_{self.name}.pickle")
        with open(path, 'wb') as outfile:
            pickle.dump(self.generator, outfile)
        json_module["module"] = path
        return json_module

    def _transform(self, input_data: xr.DataArray, logprob=False, reverse=False,
                   **kwargs: xr.DataArray) -> np.array:
        x = input_data.values.reshape((len(input_data), -1))
        conds = self._get_conditions(kwargs)

        quantiles = {}


        z = self.generator.forward(torch.Tensor(x), torch.Tensor(conds), rev=False)[0]

        noise = torch.Tensor(self.sample_size * len(x), input_data.shape[-1]).normal_(mean=1,
                                                                    std=self.std) * z.repeat(self.sample_size, 1)  # random noise around point z

        samples = self.generator.forward(noise, torch.Tensor(conds).repeat(self.sample_size, 1), rev=True)[0].detach().numpy()

        samples = samples.reshape(self.sample_size, len(x), -1)
        for k in self.quantiles:
            quantiles[k] = stats.scoreatpercentile(samples, k, axis = 0)


        arr = np.array(list(quantiles.values()))
        arr = arr.swapaxes(0, 1)
        arr = arr.swapaxes(2, 1)
        da = xr.DataArray(arr, dims=[_get_time_indexes(input_data)[0], "horizon", "quantiles"],
                          coords={"quantiles": list(quantiles.keys()),
                                  _get_time_indexes(input_data)[0]: input_data.indexes[_get_time_indexes(input_data)[0]]})
        return da

    def get_generator(self, x_features, cond_features):
        return INN(5e-4, horizon=x_features, cond_features=cond_features, n_layers_cond=10)


class INNWrapper(INNWrapperBase):

    def loss_function(self, z, log_j):
        loss = torch.mean(z ** 2) / 2 - torch.mean(log_j) / z.shape[-1]
        return loss

    def _run_epoch(self, data_loader, epoch, conds_val, x_val):
        self.generator.train()
        for batch_idx, (data, conds) in enumerate(data_loader):

            z, log_j = self.generator(data, conds)

            loss = self.loss_function(z, log_j)

            self._apply_backprop(loss)

            if not batch_idx % 50:
                with torch.no_grad():
                    z, log_j = self.generator(torch.from_numpy(x_val.astype("float32")),
                                              torch.from_numpy(conds_val.astype("float32")))
                    loss_test = self.loss_function(z, log_j)
                    print(f"{epoch}, {batch_idx}, {len(data_loader.dataset)}, {loss.item()}, {loss_test.item()}")

    def transform(self, input_data: xr.DataArray, **kwargs: Dict[str, xr.DataArray]) -> xr.DataArray:
        return self._transform(input_data=input_data, logprob=self.logprob, **kwargs)

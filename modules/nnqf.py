from typing import Dict

import xarray as xr
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pywatts.core.base import BaseTransformer

from pywatts.utils._xarray_time_series_utils import _get_time_indexes


class NNQF(BaseTransformer):

    def __init__(self, name: str="NNQF"):
        self.var_weighting = False
        self.nearest_neighbors = 100
        self.quantiles = [p / 100 for p in range(1, 100)]
        super().__init__(name)

    @staticmethod
    def _dataset_to_sklearn_input(x):
        if x is None:
            return None
        result = None
        for data_array in x.values():
            if result is not None:
                result = np.concatenate([result, data_array.values.reshape((len(data_array.values), -1))], axis=1)
            else:
                result = data_array.values.reshape((len(data_array.values), -1))
        return result

    def set_params(self, **kwargs):
        return super().set_params(**kwargs)

    def get_params(self) -> Dict[str, object]:
        return super().get_params()

    def transform(self, y, **kwargs: Dict[str, xr.DataArray]) -> xr.DataArray:

        x = self._dataset_to_sklearn_input(kwargs)
        y = y.values
        if self.var_weighting:
            var_weights = np.var(x, axis=0)
            x_input = var_weights ** (-1) * x

            # --
        # We calculate the nearest neighbor of each feature vector within the input matrix
        # and obtain their corresponding indices
        # The distance used is the minkowski distance with p = minkowski_dist

        x_neighbors = NearestNeighbors(n_neighbors=100, algorithm='auto',metric="minkowski").fit(x)
        dist, indx = x_neighbors.kneighbors(x)

        # --
        # We create a matrix containing the output values of nearest neighbors of
        # each input vector

        y_neighbors = np.take(y, indx)
        # --
        # We calculate the q_quantile of the nearest neighbors output values
        # and create with them a new output vector yq_output

        yq_output = np.quantile(y_neighbors, q=self.quantiles, axis=1).swapaxes(0,1)

        return  xr.DataArray(yq_output, dims=[_get_time_indexes(list(kwargs.values())[0])[0], "quantiles"],
                            coords={_get_time_indexes(list(kwargs.values())[0])[0]: list(kwargs.values())[0][_get_time_indexes(list(kwargs.values())[0])[0]],
                                    "quantiles": np.arange(0.01, 1, 0.01).tolist()})
    
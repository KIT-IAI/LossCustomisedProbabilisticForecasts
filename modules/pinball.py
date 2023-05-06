import logging
from typing import Dict, Callable, Optional

import xarray as xr
import numpy as np
from pywatts.core.base_summary import BaseSummary
from pywatts.core.summary_object import SummaryObjectList, SummaryObject

import pandas as pd


import tensorflow_addons as tfa
from sklearn.metrics import mean_pinball_loss

logger = logging.getLogger(__name__)


class Pinball_Loss(BaseSummary):

    def __init__(self, name: str = "Pinball_Loss",
                 quantiles: list = [50, 1, 99, 5, 95, 15, 85, 25, 75, 10, 90, 20, 80, 30, 70, 40, 60]):
        super().__init__(name)
        self.quantiles = quantiles

    def get_params(self) -> Dict[str, object]:

        return {"quantiles": self.quantiles}

    def transform(self, file_manager, y: xr.DataArray, **kwargs:xr.DataArray) -> SummaryObject:

        summary = SummaryObjectList(self.name)

        for key, y_hat in kwargs.items():
            pl = []
            for quant in y_hat.quantiles:
                pl.append(mean_pinball_loss(y.values.reshape(y.shape[:-1]),y_hat.loc[:,:,quant.values],alpha=quant.values/100))
            summary.set_kv(key, np.mean(pl))

        return summary

    def set_params(self, quantiles: Optional[list] = None):

        if quantiles:
            self.quantiles = quantiles
import logging
from typing import Dict, Callable, Optional

import xarray as xr
import numpy as np
from pywatts.core.base_summary import BaseSummary
from pywatts.core.summary_object import SummaryObjectList, SummaryObject

logger = logging.getLogger(__name__)


class Coverage_Rate(BaseSummary):

    def __init__(self, name: str = "Pinball_Loss",
                 quantiles: list = [50, 1, 99, 5, 95, 15, 85, 25, 75, 10, 90, 20, 80, 30, 70, 40, 60]):
        super().__init__(name)
        self.quantiles = quantiles

    def get_params(self) -> Dict[str, object]:

        return {"quantiles": self.quantiles}

    def transform(self, file_manager, y: xr.DataArray, **kwargs:xr.DataArray) -> SummaryObject:

        summary = SummaryObjectList(self.name)

        for key, y_hat in kwargs.items():
            p = y_hat
            t = y.values.reshape(y.shape[:-1])
            cr98_w = np.less_equal(t, p.sel(quantiles=99)) & np.greater_equal(t, p.sel(quantiles=1))
            cr98 = np.mean(cr98_w.sum(axis=1) / 24) * 100
            cr90_w = np.less_equal(t, p.sel(quantiles=95)) & np.greater_equal(t, p.sel(quantiles=5))
            cr90 = np.mean(cr90_w.sum(axis=1) / 24) * 100
            cr70_w = np.less_equal(t, p.sel(quantiles=85)) & np.greater_equal(t, p.sel(quantiles=15))
            cr70 = np.mean(cr70_w.sum(axis=1) / 24) * 100
            cr40_w = np.less_equal(t, p.sel(quantiles=70)) & np.greater_equal(t, p.sel(quantiles=30))
            cr40 = np.mean(cr40_w.sum(axis=1) / 24) * 100
            cr50_w = np.less_equal(t, p.sel(quantiles=75)) & np.greater_equal(t, p.sel(quantiles=25))
            cr50 = np.mean(cr50_w.sum(axis=1) / 24) * 100
            cr20_w = np.less_equal(t, p.sel(quantiles=60)) & np.greater_equal(t, p.sel(quantiles=40))
            cr20 = np.mean(cr20_w.sum(axis=1) / 24) * 100

            coverage_rates = dict({"98": cr98.values.mean(), "90": cr90.values.mean(), "70": cr70.values.mean(), "50": cr50.values.mean(), "40": cr40.values.mean(), "20": cr20.values.mean()})
            summary.set_kv(key, coverage_rates)

        return summary

    def set_params(self, quantiles: Optional[list] = None):

        if quantiles:
            self.quantiles = quantiles
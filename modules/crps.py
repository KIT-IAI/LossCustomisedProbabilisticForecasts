import logging
from typing import Dict, Optional

import xarray as xr
from pywatts.core.base_summary import BaseSummary

from pywatts.core.summary_object import SummaryObjectList, SummaryObject
import properscoring

logger = logging.getLogger(__name__)


class CRPS(BaseSummary):
    """
    Module to calculate the Root Mean Squared Error (RMSE)

    :param offset: Offset, which determines the number of ignored values in the beginning for calculating the RMSE.
                   Default 0
    :type offset: int
    :param rolling: Flag that determines if a rolling rmse should be used. Default False
    :type rolling: bool
    :param window: Determine the window size if a rolling rmse should be calculated. Ignored if rolling is set to
                   False. Default 24
    :type window: int

    """

    def __init__(self, name: str = "CRPS", offset: int = 0, rolling: bool = False, window: int = 24):
        super().__init__(name)
        self.offset = offset
        self.rolling = rolling
        self.window = window

    def get_params(self) -> Dict[str, object]:
        """
        Returns a dict of parameters used in the RMSE Calculator.

        :return: Parameters set for the RMSE calculator
        :rtype: Dict[str, object]
        """
        return {"offset": self.offset,
                "rolling": self.rolling,
                "window": self.window}

    def transform(self, file_manager, y: xr.DataArray, **kwargs:xr.DataArray) -> SummaryObject:

        """
        Calculates the RMSE based on the predefined target and predictions variables.

        :param x: the input dataset
        :type x: Optional[xr.DataArray]

        :return: The calculated RMSE
        :rtype: xr.DataArray
        """
        summary = SummaryObjectList(self.name)

        for key, y_hat in kwargs.items():
            crps = properscoring.crps_ensemble(y.values.reshape(y.shape[:-1]), y_hat)
            summary.set_kv(key, crps.mean())

        return summary

    def set_params(self, offset: Optional[int] = None, rolling: Optional[bool] = None, window: Optional[int] = None):
        """
        Set parameters of the RMSECalculator.

        :param offset: Offset, which determines the number of ignored values in the beginning for calculating the RMSE.
        :type offset: int
        :param rolling: Flag that determines if a rolling rmse should be used.
        :type rolling: bool
        :param window: Determine the window size if a rolling rmse should be calculated. Ignored if rolling is set to
                       False.
        :type window: int
        """
        if offset:
            self.offset = offset
        if rolling:
            self.rolling = rolling
        if window:
            self.window = window

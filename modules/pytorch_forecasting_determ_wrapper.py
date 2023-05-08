from typing import Dict
import numpy as np
import xarray as xr
from gluonts.dataset import DatasetCollection
from gluonts.dataset.common import ListDataset
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_lightning.callbacks import EarlyStopping

from pywatts.core.filemanager import FileManager
from pywatts.utils._xarray_time_series_utils import _get_time_indexes
from pywatts.modules.wrappers.dl_wrapper import DlWrapper
import pandas as pd
import pytorch_lightning as pl
class PyTorchForecastingDeterministicWrapper(DlWrapper):

    def __init__(self, model,
                 name: str = "pytorchForecastingWrapper", fit_kwargs=None):
        self.targets = []
        super().__init__(model, name, fit_kwargs)



    def fit(self, x:xr.DataArray, **kwargs: xr.DataArray):

        externals = self.split_kwargs(kwargs)

        data = pd.DataFrame(externals, columns=[str(i) for i in range(externals.shape[-1])])
        data["ts_index"] = range(len(data))
        data["value"] = x.values
        data["series"] = 0

        known_reals = set(data.columns)
        known_reals.remove("value")
        known_reals.remove("ts_index")
        known_reals.remove("series")
        self.ts_index = len(x)
        self.last = x.indexes[_get_time_indexes(x)[0]][-1]
        self.freq = x.indexes[_get_time_indexes(x)[0]][-1] - x.indexes[_get_time_indexes(x)[0]][-2]

        training = TimeSeriesDataSet(
            data,
            time_idx="ts_index",
            target="value",
            time_varying_unknown_reals=["value"],
            time_varying_known_reals=list(known_reals),
            group_ids=["series"],
            max_encoder_length=25,
            max_prediction_length=24,
        )
        validation = TimeSeriesDataSet.from_dataset(training, data,
                                                    min_prediction_idx=int(len(x) * 0.8))

        train_dataloader = training.to_dataloader(
            train=True, batch_size=32, num_workers=0
        )
        val_dataloader = validation.to_dataloader(
            train=True, batch_size=len(validation), num_workers=0
        )

        model = self.model.from_dataset(training)
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")

        trainer = pl.Trainer(
            max_epochs=350,
            gpus=0,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback],
            limit_train_batches=50,
            #   limit_val_batches=50,
            enable_checkpointing=True,
            default_root_dir="../../Results/Lightning",
        )
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        best_model_path = trainer.checkpoint_callback.best_model_path
        self.trained_model = self.model.load_from_checkpoint(best_model_path)
        self.is_fitted = True

    def get_pred_data(self, x, externals):
        dataset =  [
            ListDataset(
                [{"start": x.indexes[_get_time_indexes(x)[0]][0],
                  "target": x[:i],
                  'feat_dynamic_real': externals[:i]
                  }],
                freq="h"  # TODO more general?
            )
            for i in range(24, len(x))
        ]

        return DatasetCollection(dataset)

    def transform(self, x, **kwargs) -> xr.DataArray:

        externals = self.split_kwargs(kwargs)

        # TODO mention this bug in pytorch forecasting, int column names causes trouble
        data = pd.DataFrame(externals, columns=[str(i) for i in range(externals.shape[-1])])
        data["value"] = x.values
        data["series"] = 0

        new_last = x.indexes[_get_time_indexes(x)[0]][-1]

        nums = int((new_last - self.last) / self.freq)
        data["ts_index"] = range(self.ts_index + nums - len(data), self.ts_index + nums)
        known_reals = set(data.columns)
        known_reals.remove("value")
        known_reals.remove("ts_index")
        known_reals.remove("series")
        self.ts_index += nums

        test = TimeSeriesDataSet(
            data,
            time_idx="ts_index",
            target="value",
            time_varying_unknown_reals=["value"],
            time_varying_known_reals=list(known_reals),
            group_ids=["series"],
            max_encoder_length=25,
            max_prediction_length=24,
        )
        test_dataloader = test.to_dataloader(
            train=False, batch_size=len(test), num_workers=0
        )

        prediction = self.trained_model.predict(test_dataloader,
                                                mode="prediction")

        return xr.DataArray(prediction.numpy(), dims=["time", "horizon"],
                            coords={"time": list(kwargs.values())[0][_get_time_indexes(list(kwargs.values())[0])[0]][24:-24],
                                    "horizon": range(prediction.shape[1])})

    def save(self, fm: FileManager) -> dict:
        pass

    @classmethod
    def load(cls, load_information) -> "KerasWrapper":
        pass

    def get_params(self) -> Dict[str, object]:
        pass

    def set_params(self, fit_kwargs=None, compile_kwargs=None, custom_objects=None):
       pass

    def split_kwargs(self, kwargs):
        externals = []
        for key, value in kwargs.items():
            externals.append(value.values.reshape((len(value), -1)))
        return np.concatenate(externals, axis=-1)

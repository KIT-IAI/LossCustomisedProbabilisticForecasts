import logging
from typing import Tuple, Union, Dict

import cloudpickle
import numpy as np
import tensorflow as tf
import xarray as xr
from gluonts.dataset import DatasetCollection
from gluonts.dataset.common import ListDataset
from gluonts.dataset.pandas import PandasDataset
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_lightning.callbacks import EarlyStopping

from pywatts.core.filemanager import FileManager
from pywatts.utils._split_kwargs import split_kwargs
from pywatts.utils._xarray_time_series_utils import _get_time_indexes, xarray_to_numpy, numpy_to_xarray
from pywatts.modules.wrappers.dl_wrapper import DlWrapper
import pandas as pd
import pytorch_lightning as pl
class PyTorchForecastingDeterministicWrapper(DlWrapper):
    """
    Wrapper class for keras models

    :param model: The deep learning model
    :param name: The name of the wrappers
    :type name: str
    :param fit_kwargs: The fit keyword arguments necessary for fitting the model
    :type fit_kwargs: dict
    :param compile_kwargs: The compile keyword arguments necessary for compiling the model.
    :type compile_kwargs: dict
    :param custom_objects: This dict contains all custom objects needed by the keras model. Note,
                           users that uses such customs objects (e.g. Custom Loss) need to specify this to enable
                           the loading of the stored Keras model.
    :type custom_objects: dict
    """

    def __init__(self, model,
                 name: str = "pytorchForecastingWrapper", fit_kwargs=None):
        self.targets = []
        super().__init__(model, name, fit_kwargs)



    def fit(self, x:xr.DataArray, **kwargs: xr.DataArray):
        """
        Calls the compile and the fit method of the wrapped keras module.
        :param x: The input data
        :param y: The target data
        """

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
        """
        Calls predict of the underlying keras Model.
        :param x: The dataset for which a prediction should be performed
        :return:  The prediction. Each output of the keras model is a separate data variable in the returned xarray.
        """
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
        """
        Stores the keras model at the given path
        :param fm: The Filemanager, which contains the path where the model should be stored
        :return: The path where the model is stored.
        """
        #TODO: Fix so that pytorch models are saved
        json = {"name": self.name,
                "class": self.__class__.__name__,
                "module": self.__module__}

        params = self.get_params()
        params_path = fm.get_path(f"{self.name}_params.pickle")
        with open(params_path, "wb") as outfile:
            cloudpickle.dump(params, outfile)
        json["params"] = params_path
        json["is_fitted"] = self.is_fitted

        custom_path = fm.get_path(f"{self.name}_custom.pickle")
        with open(custom_path, "wb") as outfile:
            cloudpickle.dump(self.custom_objects, outfile)
        json["custom_objects"] = custom_path

        json["targets"] = self.targets
        model_path = fm.get_path(f"{self.name}.h5")
        self.model.save(filepath=model_path)
        aux_models = []
        for name, aux_model in self.aux_models.items():
            aux_model_path = fm.get_path(f"{self.name}_{name}.h5")
            aux_model.save(filepath=aux_model_path)
            aux_models.append((name, aux_model_path))
        json.update({
            "aux_models": aux_models,
            "model": model_path
        })

        return json

    @classmethod
    def load(cls, load_information) -> "KerasWrapper":
        """
        Load the keras model and instantiate a new keraswrapper class containing the model.
        :param params:  The paramters which should be used for restoring the model.
        (Note: This models should be taken from the pipeline json file)
        :return: A wrapped keras model.
        """
        #TODO: fix so that pytorch models are loaded
        name = load_information["name"]
        params_path = load_information["params"]
        with open(params_path, "rb") as infile:
            params = cloudpickle.load(infile)

        custom_path = load_information["custom_objects"]
        with open(custom_path, "rb") as infile:
            custom_objects = cloudpickle.load(infile)

        try:
            model = tf.keras.models.load_model(filepath=load_information["model"], custom_objects=custom_objects)
        except Exception as exception:
            logging.error("No model found in %s.", load_information['model'])
            raise exception
        aux_models = {}
        if "aux_models" in load_information.keys():
            for aux_name, path in load_information["aux_models"]:
                try:
                    aux_models[aux_name] = tf.keras.models.load_model(filepath=path, custom_objects=custom_objects)
                except Exception as exception:
                    logging.error("No model found in path %s", path)
                    raise exception
            module = cls((model, aux_models), name=name, **params)
        else:
            module = cls(model, name=name, **params)
        module.is_fitted = load_information["is_fitted"]

        module.targets = load_information["targets"]
        return module

    def get_params(self) -> Dict[str, object]:
        """
        Returns the parameters of deep learning frameworks.
        :return: A dict containing the fit keyword arguments and the compile keyword arguments
        """
        #Todo: fix so that pytorch models parameters are returned
        return {
            "fit_kwargs": self.fit_kwargs,
            "compile_kwargs": self.compile_kwargs,
            "custom_objects": self.custom_objects
        }

    def set_params(self, fit_kwargs=None, compile_kwargs=None, custom_objects=None):
        """
        Set the parameters of the deep learning wrappers
        :param fit_kwargs: keyword arguments for the fit method.
        :param compile_kwargs: keyword arguments for the compile methods.
        :param custom_objects: This dict contains all custom objects needed by the keras model. Note,
                               users that uses such customs objects (e.g. Custom Loss) need to specify this to enable
                               the loading of the stored Keras model.
        """
        #Todo: fix so that pytorch models parmaeters are set
        if fit_kwargs:
            self.fit_kwargs = fit_kwargs
        if compile_kwargs:
            self.compile_kwargs = compile_kwargs
        if custom_objects:
            self.custom_objects = custom_objects

    def split_kwargs(self, kwargs):
        externals = []
        for key, value in kwargs.items():
            externals.append(value.values.reshape((len(value), -1)))
        return np.concatenate(externals, axis=-1)

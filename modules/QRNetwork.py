
from typing import Dict

import numpy as np
import tensorflow as tf
import xarray as xr
import keras.layers as layers
from keras.models import Model

from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from pywatts.core.base import BaseTransformer
import keras.backend as K

from pywatts.core.filemanager import FileManager
from pywatts.utils._xarray_time_series_utils import _get_time_indexes



def mcycleModel(number_of_features):

    if number_of_features == 0:
        input_1 = layers.Input(shape=(24,), name='history')
        input_2 = layers.Input(shape=(24, 5,), name='calendar')
        flatten = layers.Flatten()(input_2)
        merged = layers.Concatenate(axis=1)([input_1, flatten])
        hidden_1 = layers.Dense(90, activation="relu", name="hidden_1")(merged)
        hidden_2 = layers.Dense(64, activation="relu", name="hidden_2")(hidden_1)
        hidden_3 = layers.Dense(32, activation="relu", name="hidden_3")(hidden_2)
        output = layers.Dense(24, activation='linear', name='target')(hidden_3)
        model = Model(inputs=[input_1, input_2], outputs=output)

    if number_of_features == 1:
        input_1 = layers.Input(shape=(24,), name='history')
        input_2 = layers.Input(shape=(24, 5,), name='calendar')
        input_3 = layers.Input(shape=(24,), name="feature1")
        flatten = layers.Flatten()(input_2)
        merged = layers.Concatenate(axis=1)([input_1, flatten, input_3])
        hidden_1 = layers.Dense(90, activation="relu", name="hidden_1")(merged)
        hidden_2 = layers.Dense(64, activation="relu", name="hidden_2")(hidden_1)
        hidden_3 = layers.Dense(32, activation="relu", name="hidden_3")(hidden_2)
        output = layers.Dense(24, activation='linear', name='target')(hidden_3)
        model = Model(inputs=[input_1, input_2, input_3], outputs=output)

    if number_of_features == 2:
        input_1 = layers.Input(shape=(24,), name='history')
        input_2 = layers.Input(shape=(24, 5,), name='calendar')
        input_3 = layers.Input(shape=(24,), name="feature1")
        input_4 = layers.Input(shape=(24,), name="feature2")
        flatten = layers.Flatten()(input_2)
        merged = layers.Concatenate(axis=1)([input_1, flatten, input_3, input_4])
        hidden_1 = layers.Dense(90, activation="relu", name="hidden_1")(merged)
        hidden_2 = layers.Dense(64, activation="relu", name="hidden_2")(hidden_1)
        hidden_3 = layers.Dense(32, activation="relu", name="hidden_3")(hidden_2)
        output = layers.Dense(24, activation='linear', name='target')(hidden_3)
        model = Model(inputs=[input_1, input_2, input_3, input_4], outputs=output)

    if number_of_features == 3:
        input_1 = layers.Input(shape=(24,), name='history')
        input_2 = layers.Input(shape=(24, 5,), name='calendar')
        input_3 = layers.Input(shape=(24,), name="feature1")
        input_4 = layers.Input(shape=(24,), name="feature2")
        input_5 = layers.Input(shape=(24,), name="feature3")
        flatten = layers.Flatten()(input_2)
        merged = layers.Concatenate(axis=1)([input_1, flatten, input_3, input_4, input_5])
        hidden_1 = layers.Dense(90, activation="relu", name="hidden_1")(merged)
        hidden_2 = layers.Dense(64, activation="relu", name="hidden_2")(hidden_1)
        hidden_3 = layers.Dense(32, activation="relu", name="hidden_3")(hidden_2)
        output = layers.Dense(24, activation='linear', name='target')(hidden_3)
        model = Model(inputs=[input_1, input_2, input_3, input_4, input_5], outputs=output)

    if number_of_features == 4:
        input_1 = layers.Input(shape=(24,), name='history')
        input_2 = layers.Input(shape=(24, 5,), name='calendar')
        input_3 = layers.Input(shape=(24,), name="feature1")
        input_4 = layers.Input(shape=(24,), name="feature2")
        input_5 = layers.Input(shape=(24,), name="feature3")
        input_6 = layers.Input(shape=(24,), name="feature4")
        flatten = layers.Flatten()(input_2)
        merged = layers.Concatenate(axis=1)([input_1, flatten, input_3, input_4, input_5, input_6])
        hidden_1 = layers.Dense(90, activation="relu", name="hidden_1")(merged)
        hidden_2 = layers.Dense(64, activation="relu", name="hidden_2")(hidden_1)
        hidden_3 = layers.Dense(32, activation="relu", name="hidden_3")(hidden_2)
        output = layers.Dense(24, activation='linear', name='target')(hidden_3)
        model = Model(inputs=[input_1, input_2, input_3, input_4, input_5, input_6], outputs=output)

    return model

def tilted_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

class PLNN(BaseTransformer):


    def __init__(self, name="name",
                 return_type="quantiles",
                 learning_rate=0.01, store_path="",
                 early_stopping_loss = "val_loss", q_values=[0.01,0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.96,0.97,0.98,.99],
                 number_of_features=0):
        super().__init__(name)
        self.return_type = return_type
        self.has_predict_proba=True
        self.opti = Adam(learning_rate=learning_rate)
        self.store_path =store_path
        self.early_stopping_loss = early_stopping_loss
        self.best_hyperparams = {}
        self.q_values=q_values
        self.number_of_features=number_of_features

    def fit(self,
            history,
            calendar,
            target,
            feature1=None,
            feature2=None,
            feature3=None,
            feature4=None,
            feature5=None,
            feature6=None,
            **kwargs
            ):

        if self.number_of_features==0:
            x ={
                "history":history.values, "calendar":calendar.values
                }
        if self.number_of_features==1:
            x ={
                "history":history.values, "calendar":calendar.values, "feature1":feature1.values
                }
        if self.number_of_features==2:
            x ={
                "history":history.values, "calendar":calendar.values, "feature1":feature1.values, "feature2":feature2.values
                }
        if self.number_of_features==3:
            x ={
                "history":history.values, "calendar":calendar.values, "feature1":feature1.values, "feature2":feature2.values,
                "feature3":feature3.values
                }
        if self.number_of_features==4:
            x ={
                "history":history.values, "calendar":calendar.values, "feature1":feature1.values, "feature2":feature2.values,
                "feature3":feature3.values, "feature4":feature4.values
                }

        self.models = []
        for q in self.q_values:
            model = mcycleModel(self.number_of_features)
            model.compile(loss=lambda y,f: tilted_loss(q,y,f), optimizer='adam')
            model.fit(
                x=x, y={"target":target.values}, **{"epochs":100, "validation_split":0.2,
                                                          "batch_size":32, "verbose":1,
                                                          "callbacks":[EarlyStopping("val_loss",
                                                                                     patience=5,
                                                                                     restore_best_weights=True),
                                                                       tf.keras.callbacks.TensorBoard(
                                                                           log_dir='logs',
                                                                           histogram_freq=0,
                                                                           write_graph=True,
                                                                           write_images=False,
                                                                           write_steps_per_second=False,
                                                                           update_freq='epoch',
                                                                           profile_batch=0,
                                                                           embeddings_freq=0,
                                                                           embeddings_metadata=None,
                                                                       )
                                                                       ]})
            self.models.append(model)
        self.is_fitted=True

    def transform(self, **kwargs: xr.DataArray) -> xr.DataArray:
        quantiles = []

        for model in self.models:
            quantiles.append(model.predict({key: da.values for key, da in kwargs.items()}))

        quantiles = np.stack(quantiles, -1)
        quantiles.sort()

        quant_labels = [int(i * 100) for i in self.q_values]
        quant_labels.sort()

        return xr.DataArray(quantiles, dims=["time", "horizon", "quantiles"],
         coords={"time":list(kwargs.values())[0][_get_time_indexes(list(kwargs.values())[0])[0]].values,
                 "horizon": range(quantiles.shape[1]),
                 "quantiles": quant_labels})


    def save(self, fm: FileManager) -> dict:
        pass

    @classmethod
    def load(cls, load_information) -> "KerasWrapper":
        pass

    def get_params(self) -> Dict[str, object]:
        pass

    def set_params(self, fit_kwargs=None, compile_kwargs=None, custom_objects=None):
        pass
import logging

from hyperopt import hp

from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from pywatts.summaries import RMSE, MAE
from pywatts.callbacks.npy_callback import NPYCallback
from modules.pinball import Pinball_Loss
from pywatts.modules import ClockShift, SKLearnWrapper, Sampler, FunctionModule, Slicer

import keras.layers as layers
from keras.models import Model


from sklearn.preprocessing import StandardScaler

from modules.crps import CRPS
from modules.CR import Coverage_Rate
from modules.generator_base import GeneratorBase
from modules.cinn_prob_forecaster import ProbForecastCINN

from modules.hyperparameter_optimization import RayTuneWrapper, SplitMethod
from ray import tune, air
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.stopper import TimeoutStopper, CombinedStopper, ExperimentPlateauStopper


def train_forecast_pipeline(pipeline_name, target, sklearn_estimators, pytorch_estimators, calendar_extraction, features):
    pipeline = Pipeline(f"../../Results/{pipeline_name}/train_forecaster", name="train_forecaster")

    # Extract calendar features
    calendar = calendar_extraction(x=pipeline[target])
    calendar_sample = Sampler(24, name="CalendarSample")(x=calendar)
    calendar_sample = Slicer(start=2 * 24, end=-2 * 24)(x=calendar_sample)
    calendar_slice = ClockShift(lag=24)(x=calendar)
    calendar_slice = Slicer(start=24, end=-24)(x=calendar_slice)

    # Scale the target
    target_scaler = SKLearnWrapper(module=StandardScaler(), name="target_scaler")
    scale_target = target_scaler(x=pipeline[target])

    # Sample target
    target_sample = Sampler(24, name=f"Target_Sample")(x=scale_target)
    target_sample = Slicer(start=2 * 24, end=-2 * 24)(x=target_sample)
    target_slice = ClockShift(lag=24)(x=scale_target)
    target_slice = Slicer(start=24, end=-24)(x=target_slice)

    # Create Historical Values and Sample
    target_history = ClockShift(lag=25, name="History_Target")(x=scale_target)
    history_sample = Sampler(24, name="History_Sample")(x=target_history)
    history_sample = Slicer(start=2 * 24, end=-2 * 24)(x=history_sample)

    # Deal with features
    number_of_features = len(features)
    feature1_scaler = None
    feature2_scaler = None
    feature3_scaler = None
    feature4_scaler = None

    if number_of_features == 0:
        for sklearn_estimator in sklearn_estimators:
            sklearn_estimator(history=history_sample,
                              calendar=calendar_sample,
                              target=target_sample)

        for pytorch_estimator in pytorch_estimators:
            pytorch_estimator(x=target_slice,
                              calendar=calendar_slice)
    if number_of_features == 1:
        feature1 = features[0]
        feature1_scaler = SKLearnWrapper(module=StandardScaler(), name="feature1_scaler")
        scale_feature1 = feature1_scaler(x=pipeline[feature1])
        feature1_sample = Sampler(24, name="Feature1_Sample")(x=scale_feature1)
        feature1_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature1_sample)
        feature1_slice = ClockShift(lag=24)(x=scale_feature1)
        feature1_slice = Slicer(start=24, end=-24)(x=feature1_slice)
        for sklearn_estimator in sklearn_estimators:
            sklearn_estimator(history=history_sample,
                              calendar=calendar_sample,
                              feature1=feature1_sample,
                              target=target_sample)

        for pytorch_estimator in pytorch_estimators:
            pytorch_estimator(x=target_slice,
                              calendar=calendar_slice,
                              feature1=feature1_slice)
    if number_of_features == 2:
        feature1 = features[0]
        feature1_scaler = SKLearnWrapper(module=StandardScaler(), name="feature1_scaler")
        scale_feature1 = feature1_scaler(x=pipeline[feature1])
        feature1_sample = Sampler(24, name="Feature1_Sample")(x=scale_feature1)
        feature1_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature1_sample)
        feature1_slice = ClockShift(lag=24)(x=scale_feature1)
        feature1_slice = Slicer(start=24, end=-24)(x=feature1_slice)
        feature2 = features[1]
        feature2_scaler = SKLearnWrapper(module=StandardScaler(), name="feature2_scaler")
        scale_feature2 = feature2_scaler(x=pipeline[feature2])
        feature2_sample = Sampler(24, name="Feature2_Sample")(x=scale_feature2)
        feature2_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature2_sample)
        feature2_slice = ClockShift(lag=24)(x=scale_feature2)
        feature2_slice = Slicer(start=24, end=-24)(x=feature2_slice)
        for sklearn_estimator in sklearn_estimators:
            sklearn_estimator(history=history_sample,
                              calendar=calendar_sample,
                              feature1=feature1_sample,
                              feature2=feature2_sample,
                              target=target_sample)

        for pytorch_estimator in pytorch_estimators:
            pytorch_estimator(x=target_slice,
                              calendar=calendar_slice,
                              feature1=feature1_slice,
                              feature2=feature2_slice)
    if number_of_features == 3:
        feature1 = features[0]
        feature1_scaler = SKLearnWrapper(module=StandardScaler(), name="feature1_scaler")
        scale_feature1 = feature1_scaler(x=pipeline[feature1])
        feature1_sample = Sampler(24, name="Feature1_Sample")(x=scale_feature1)
        feature1_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature1_sample)
        feature1_slice = ClockShift(lag=24)(x=scale_feature1)
        feature1_slice = Slicer(start=24, end=-24)(x=feature1_slice)
        feature2 = features[1]
        feature2_scaler = SKLearnWrapper(module=StandardScaler(), name="feature2_scaler")
        scale_feature2 = feature2_scaler(x=pipeline[feature2])
        feature2_sample = Sampler(24, name="Feature2_Sample")(x=scale_feature2)
        feature2_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature2_sample)
        feature2_slice = ClockShift(lag=24)(x=scale_feature2)
        feature2_slice = Slicer(start=24, end=-24)(x=feature2_slice)
        feature3 = features[2]
        feature3_scaler = SKLearnWrapper(module=StandardScaler(), name="feature3_scaler")
        scale_feature3 = feature3_scaler(x=pipeline[feature3])
        feature3_sample = Sampler(24, name="Feature3_Sample")(x=scale_feature3)
        feature3_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature3_sample)
        feature3_slice = ClockShift(lag=24)(x=scale_feature3)
        feature3_slice = Slicer(start=24, end=-24)(x=feature3_slice)
        for sklearn_estimator in sklearn_estimators:
            sklearn_estimator(history=history_sample,
                              calendar=calendar_sample,
                              feature1=feature1_sample,
                              feature2=feature2_sample,
                              feature3=feature3_sample,
                              target=target_sample)

        for pytorch_estimator in pytorch_estimators:
            pytorch_estimator(x=target_slice,
                              calendar=calendar_slice,
                              feature1=feature1_slice,
                              feature2=feature2_slice,
                              feature3=feature3_slice)
    if number_of_features == 4:
        feature1 = features[0]
        feature1_scaler = SKLearnWrapper(module=StandardScaler(), name="feature1_scaler")
        scale_feature1 = feature1_scaler(x=pipeline[feature1])
        feature1_sample = Sampler(24, name="Feature1_Sample")(x=scale_feature1)
        feature1_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature1_sample)
        feature1_slice = ClockShift(lag=24)(x=scale_feature1)
        feature1_slice = Slicer(start=24, end=-24)(x=feature1_slice)
        feature2 = features[1]
        feature2_scaler = SKLearnWrapper(module=StandardScaler(), name="feature2_scaler")
        scale_feature2 = feature2_scaler(x=pipeline[feature2])
        feature2_sample = Sampler(24, name="Feature2_Sample")(x=scale_feature2)
        feature2_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature2_sample)
        feature2_slice = ClockShift(lag=24)(x=scale_feature2)
        feature2_slice = Slicer(start=24, end=-24)(x=feature2_slice)
        feature3 = features[2]
        feature3_scaler = SKLearnWrapper(module=StandardScaler(), name="feature3_scaler")
        scale_feature3 = feature3_scaler(x=pipeline[feature3])
        feature3_sample = Sampler(24, name="Feature3_Sample")(x=scale_feature3)
        feature3_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature3_sample)
        feature3_slice = ClockShift(lag=24)(x=scale_feature3)
        feature3_slice = Slicer(start=24, end=-24)(x=feature3_slice)
        feature4 = features[3]
        feature4_scaler = SKLearnWrapper(module=StandardScaler(), name="feature4_scaler")
        scale_feature4 = feature4_scaler(x=pipeline[feature4])
        feature4_sample = Sampler(24, name="Feature4_Sample")(x=scale_feature4)
        feature4_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature4_sample)
        feature4_slice = ClockShift(lag=24)(x=scale_feature4)
        feature4_slice = Slicer(start=24, end=-24)(x=feature4_slice)
        for sklearn_estimator in sklearn_estimators:
            sklearn_estimator(history=history_sample,
                              calendar=calendar_sample,
                              feature1=feature1_sample,
                              feature2=feature2_sample,
                              feature3=feature3_sample,
                              feature4=feature4_sample,
                              target=target_sample)

        for pytorch_estimator in pytorch_estimators:
            pytorch_estimator(x=target_slice,
                              calendar=calendar_slice,
                              feature1=feature1_slice,
                              feature2=feature2_slice,
                              feature3=feature3_slice,
                              feature4=feature4_slice)
    if number_of_features >= 5:
        raise ValueError("Error: Features not considered") from Exception(
            "Currently a maximum of four features are supported")

    return pipeline, target_scaler, feature1_scaler, feature2_scaler, feature3_scaler, feature4_scaler


def train_cinn_pipeline(pipeline_name, target, calendar_extraction, cinn: GeneratorBase, cinn_epochs, features):
    pipeline = Pipeline(f"../../Results/{pipeline_name}/train_cinn", name=f"train_cinn")

    # Extract calendar features
    calendar = calendar_extraction(x=pipeline[target])
    calendar_sample = Sampler(24, name="CalendarSample")(x=calendar)
    calendar_sample = Slicer(start=2 * 24, end=-2 * 24)(x=calendar_sample)

    # Scale the target
    target_scaler = SKLearnWrapper(module=StandardScaler(), name="target_scaler")
    scale_target = target_scaler(x=pipeline[target])

    # Sample target
    target_sample = Sampler(24, name=f"Target_Sample")(x=scale_target)
    target_sample = Slicer(start=2 * 24, end=-2 * 24)(x=target_sample)

    # Create Historical Values and Sample
    target_history = ClockShift(lag=25, name="History_Target")(x=scale_target)
    history_sample = Sampler(24, name="History_Sample")(x=target_history)
    history_sample = Slicer(start=2 * 24, end=-2 * 24)(x=history_sample)

    # Config cINN
    cinn.set_params(epochs=cinn_epochs)

    # Deal with features
    number_of_features = len(features)
    feature1_scaler = None
    feature2_scaler = None
    feature3_scaler = None
    feature4_scaler = None

    if number_of_features == 0:
        cinn(history=history_sample,
             calendar=calendar_sample,
             input_data=target_sample)
    if number_of_features == 1:
        feature1 = features[0]
        feature1_scaler = SKLearnWrapper(module=StandardScaler(), name="feature1_scaler")
        scale_feature1 = feature1_scaler(x=pipeline[feature1])
        feature1_sample = Sampler(24, name="Feature1_Sample")(x=scale_feature1)
        feature1_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature1_sample)
        cinn(history=history_sample,
             calendar=calendar_sample,
             feature1=feature1_sample,
             input_data=target_sample)
    if number_of_features == 2:
        feature1 = features[0]
        feature1_scaler = SKLearnWrapper(module=StandardScaler(), name="feature1_scaler")
        scale_feature1 = feature1_scaler(x=pipeline[feature1])
        feature1_sample = Sampler(24, name="Feature1_Sample")(x=scale_feature1)
        feature1_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature1_sample)
        feature2 = features[1]
        feature2_scaler = SKLearnWrapper(module=StandardScaler(), name="feature2_scaler")
        scale_feature2 = feature2_scaler(x=pipeline[feature2])
        feature2_sample = Sampler(24, name="Feature2_Sample")(x=scale_feature2)
        feature2_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature2_sample)
        cinn(history=history_sample,
             calendar=calendar_sample,
             feature1=feature1_sample,
             feature2=feature2_sample,
             input_data=target_sample)
    if number_of_features == 3:
        feature1 = features[0]
        feature1_scaler = SKLearnWrapper(module=StandardScaler(), name="feature1_scaler")
        scale_feature1 = feature1_scaler(x=pipeline[feature1])
        feature1_sample = Sampler(24, name="Feature1_Sample")(x=scale_feature1)
        feature1_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature1_sample)
        feature2 = features[1]
        feature2_scaler = SKLearnWrapper(module=StandardScaler(), name="feature2_scaler")
        scale_feature2 = feature2_scaler(x=pipeline[feature2])
        feature2_sample = Sampler(24, name="Feature2_Sample")(x=scale_feature2)
        feature2_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature2_sample)
        feature3 = features[2]
        feature3_scaler = SKLearnWrapper(module=StandardScaler(), name="feature3_scaler")
        scale_feature3 = feature3_scaler(x=pipeline[feature3])
        feature3_sample = Sampler(24, name="Feature3_Sample")(x=scale_feature3)
        feature3_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature3_sample)
        cinn(history=history_sample,
             calendar=calendar_sample,
             feature1=feature1_sample,
             feature2=feature2_sample,
             feature3=feature3_sample,
             input_data=target_sample)
    if number_of_features == 4:
        feature1 = features[0]
        feature1_scaler = SKLearnWrapper(module=StandardScaler(), name="feature1_scaler")
        scale_feature1 = feature1_scaler(x=pipeline[feature1])
        feature1_sample = Sampler(24, name="Feature1_Sample")(x=scale_feature1)
        feature1_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature1_sample)
        feature2 = features[1]
        feature2_scaler = SKLearnWrapper(module=StandardScaler(), name="feature2_scaler")
        scale_feature2 = feature2_scaler(x=pipeline[feature2])
        feature2_sample = Sampler(24, name="Feature2_Sample")(x=scale_feature2)
        feature2_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature2_sample)
        feature3 = features[2]
        feature3_scaler = SKLearnWrapper(module=StandardScaler(), name="feature3_scaler")
        scale_feature3 = feature3_scaler(x=pipeline[feature3])
        feature3_sample = Sampler(24, name="Feature3_Sample")(x=scale_feature3)
        feature3_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature3_sample)
        feature4 = features[3]
        feature4_scaler = SKLearnWrapper(module=StandardScaler(), name="feature4_scaler")
        scale_feature4 = feature4_scaler(x=pipeline[feature4])
        feature4_sample = Sampler(24, name="Feature4_Sample")(x=scale_feature4)
        feature4_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature4_sample)
        cinn(history=history_sample,
             calendar=calendar_sample,
             feature1=feature1_sample,
             feature2=feature2_sample,
             feature3=feature3_sample,
             feature4=feature4_sample,
             input_data=target_sample)
    if number_of_features >= 5:
        raise ValueError("Error: Features not considered") from Exception(
            "Currently a maximum of four features are supported")

    return pipeline, cinn


def get_prob_forecast_pipeline(pipeline_name, target, calendar_extraction, target_scaler, sklearn_estimators, pytorch_estimators,
                               cinn_base: GeneratorBase, cinn_quantiles, cinn_sample_size, cinn_sampling_std,
                               cinn_sampling_lower, cinn_sampling_upper, tuner_loss_metric, tuner_timeout, features,
                               feature1_scaler, feature2_scaler, feature3_scaler, feature4_scaler):
    pipeline = Pipeline(f"../../Results/{pipeline_name}/probabilistic_forecast_{tuner_loss_metric.name}",
                        name=f"prob_forecast_{tuner_loss_metric.name}")

    # Extract calendar features
    calendar = calendar_extraction(x=pipeline[target])
    calendar_sample = Sampler(24, name="CalendarSample")(x=calendar)
    calendar_sample = Slicer(start=2 * 24, end=-2 * 24)(x=calendar_sample)
    calendar_slice = ClockShift(lag=24)(x=calendar)
    calendar_slice = Slicer(start=24, end=-24)(x=calendar_slice)

    # Scale the target
    scale_target = target_scaler(x=pipeline[target])
    rescaled_target = target_scaler(x=scale_target,
                                    computation_mode=ComputationMode.Transform,
                                    use_inverse_transform=True)

    # Sample target
    target_sample = Sampler(24, name=f"Target_Sample")(x=scale_target)
    target_sample = Slicer(start=2 * 24, end=-2 * 24)(x=target_sample)
    rescaled_target_sample = Sampler(24, name=f"Rescaled_Target_Sample")(x=rescaled_target)
    rescaled_target_sample = Slicer(start=2 * 24, end=-2 * 24)(x=rescaled_target_sample,
                                                               callbacks=[NPYCallback(f'Target')])
    target_slice = ClockShift(lag=24)(x=scale_target)
    target_slice = Slicer(start=24, end=-24)(x=target_slice)

    # Create Historical Values and Sample
    target_history = ClockShift(lag=25, name="History_Target")(x=scale_target)
    history_sample = Sampler(24, name="History_Sample")(x=target_history)
    history_sample = Slicer(start=2 * 24, end=-2 * 24)(x=history_sample)

    # Deal with features
    number_of_features = len(features)

    # Save Results
    scaled_results_dict = {}
    final_results_dict = {}
    scaled_derministic_results = {}

    if number_of_features == 0:
        for sklearn_estimator in sklearn_estimators:
            deterministic_forecast = sklearn_estimator(history=history_sample,
                                                       calendar=calendar_sample,
                                                       computation_mode=ComputationMode.Transform)
            scaled_derministic_results[sklearn_estimator.name] = deterministic_forecast

            prob_forecaster = ProbForecastCINN(name=f"{sklearn_estimator.name}_{tuner_loss_metric.name}_prob_forecast",
                                               cinn=cinn_base,
                                               quantiles=cinn_quantiles,
                                               sample_size=cinn_sample_size,
                                               sampling_std=cinn_sampling_std)



            search_space = {
                "sampling_std": hp.uniform("sampling_std", cinn_sampling_lower, cinn_sampling_upper)
            }

            ray_tune_kwargs = {
                "tune_config": tune.TuneConfig(
                    search_alg=HyperOptSearch(space=search_space,
                                              metric="loss", mode="min",
                                              random_state_seed=314),
                    max_concurrent_trials=8,
                    num_samples=-1,  # -1 evaluates infinite configuration samples until time budget is exhausted
                    # time_budget_s is not reset, use run_config
                ),
                "run_config": air.RunConfig(
                    local_dir=f"../../Results/{pipeline_name}/prob_hpo",
                    stop=CombinedStopper(
                        TimeoutStopper(timeout=tuner_timeout),
                        ExperimentPlateauStopper(metric="loss", mode="min", std=0.001, top=10, patience=15 * 40)),
                    # verbose=0
                )
            }

            ray_init_kwargs = {
                #"local_mode": True,  # for debugging
                "log_to_driver": False,
                "logging_level": logging.FATAL
            }

            tuner = RayTuneWrapper(estimator=prob_forecaster, val_split=1, cv=5, split_method=SplitMethod.FixedOrigin, loss_metric=tuner_loss_metric,
                                   ray_tune_kwargs=ray_tune_kwargs, ray_init_kwargs=ray_init_kwargs, prob_model=True, name=f"Tuner_{sklearn_estimator.name}_{tuner_loss_metric.name}")

            prob_forecast = tuner(input_data=deterministic_forecast,
                                  history=history_sample,
                                  calendar=calendar_sample,
                                  target=target_sample,
                                  config_summary=["k_best_config_"]
                                  )
            scaled_results_dict[sklearn_estimator.name] = prob_forecast

            rescaled_forecast = target_scaler(x=prob_forecast,
                                              computation_mode=ComputationMode.Transform,
                                              use_inverse_transform=True)

            rescaled_forecast = FunctionModule(correct_shape)(sf=prob_forecast, rf=rescaled_forecast,
                                                              callbacks=[NPYCallback(
                                                                  f'Probabilistic_Forecast_{sklearn_estimator.name}_{tuner_loss_metric.name}')])

            final_results_dict[sklearn_estimator.name] = rescaled_forecast

        for pytorch_estimator in pytorch_estimators:
            deterministic_forecast = pytorch_estimator(x=target_slice,
                                                       calendar=calendar_slice,
                                                       computation_mode=ComputationMode.Transform)
            scaled_derministic_results[pytorch_estimator.name] = deterministic_forecast

            prob_forecaster = ProbForecastCINN(name=f"{pytorch_estimator.name}_{tuner_loss_metric.name}_prob_forecast",
                                               cinn=cinn_base,
                                               quantiles=cinn_quantiles,
                                               sample_size=cinn_sample_size,
                                               sampling_std=cinn_sampling_std)

            search_space = {
                "sampling_std": hp.uniform("sampling_std", cinn_sampling_lower, cinn_sampling_upper)
            }

            ray_tune_kwargs = {
                "tune_config": tune.TuneConfig(
                    search_alg=HyperOptSearch(space=search_space,
                                              metric="loss", mode="min",
                                              random_state_seed=314),
                    max_concurrent_trials=8,
                    num_samples=-1,  # -1 evaluates infinite configuration samples until time budget is exhausted
                    # time_budget_s is not reset, use run_config
                ),
                "run_config": air.RunConfig(
                    local_dir=f"../../Results/{pipeline_name}/prob_hpo",
                    stop=CombinedStopper(
                        TimeoutStopper(timeout=tuner_timeout),
                        ExperimentPlateauStopper(metric="loss", mode="min", std=0.001, top=10, patience=15 * 40)),
                    # verbose=0
                )
            }

            ray_init_kwargs = {
                #"local_mode": True,  # for debugging
                "log_to_driver": False,
                "logging_level": logging.FATAL
            }

            tuner = RayTuneWrapper(estimator=prob_forecaster, val_split=1, cv=5, split_method=SplitMethod.FixedOrigin, loss_metric=tuner_loss_metric,
                                   ray_tune_kwargs=ray_tune_kwargs, ray_init_kwargs=ray_init_kwargs, prob_model=True, name=f"Tuner_{pytorch_estimator.name}_{tuner_loss_metric.name}")

            prob_forecast = tuner(input_data=deterministic_forecast,
                                  history=history_sample,
                                  calendar=calendar_sample,
                                  target=target_sample,
                                  config_summary=["k_best_config_"]
                                  )
            scaled_results_dict[pytorch_estimator.name] = prob_forecast

            rescaled_forecast = target_scaler(x=prob_forecast,
                                              computation_mode=ComputationMode.Transform,
                                              use_inverse_transform=True)

            rescaled_forecast = FunctionModule(correct_shape)(sf=prob_forecast, rf=rescaled_forecast,
                                                              callbacks=[NPYCallback(
                                                                  f'Probabilistic_Forecast_{pytorch_estimator.name}_{tuner_loss_metric.name}')])

            final_results_dict[pytorch_estimator.name] = rescaled_forecast

    if number_of_features == 1:
        feature1 = features[0]
        scale_feature1 = feature1_scaler(x=pipeline[feature1])
        feature1_sample = Sampler(24, name="Feature1_Sample")(x=scale_feature1)
        feature1_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature1_sample)
        feature1_slice = ClockShift(lag=24)(x=scale_feature1)
        feature1_slice = Slicer(start=24, end=-24)(x=feature1_slice)

        for sklearn_estimator in sklearn_estimators:
            deterministic_forecast = sklearn_estimator(history=history_sample,
                                                       calendar=calendar_sample,
                                                       feature1=feature1_sample,
                                                       computation_mode=ComputationMode.Transform)
            scaled_derministic_results[sklearn_estimator.name] = deterministic_forecast

            prob_forecaster = ProbForecastCINN(name=f"{sklearn_estimator.name}_{tuner_loss_metric.name}_prob_forecast",
                                               cinn=cinn_base,
                                               quantiles=cinn_quantiles,
                                               sample_size=cinn_sample_size,
                                               sampling_std=cinn_sampling_std)

            search_space = {
                "sampling_std": hp.uniform("sampling_std", cinn_sampling_lower, cinn_sampling_upper)
            }

            ray_tune_kwargs = {
                "tune_config": tune.TuneConfig(
                    search_alg=HyperOptSearch(space=search_space,
                                              metric="loss", mode="min",
                                              random_state_seed=314),
                    max_concurrent_trials=8,
                    num_samples=-1,  # -1 evaluates infinite configuration samples until time budget is exhausted
                    # time_budget_s is not reset, use run_config
                ),
                "run_config": air.RunConfig(
                    local_dir=f"../../Results/{pipeline_name}/prob_hpo",
                    stop=CombinedStopper(
                        TimeoutStopper(timeout=tuner_timeout),
                        ExperimentPlateauStopper(metric="loss", mode="min", std=0.001, top=10, patience=15 * 40)),
                    # verbose=0
                )
            }

            ray_init_kwargs = {
                # "local_mode": True,  # for debugging
                "log_to_driver": False,
                "logging_level": logging.FATAL
            }

            tuner = RayTuneWrapper(estimator=prob_forecaster, val_split=1,  cv=5, split_method=SplitMethod.FixedOrigin, loss_metric=tuner_loss_metric,
                                   ray_tune_kwargs=ray_tune_kwargs, ray_init_kwargs=ray_init_kwargs, prob_model=True, name=f"Tuner_{sklearn_estimator.name}_{tuner_loss_metric.name}")

            prob_forecast = tuner(input_data=deterministic_forecast,
                                  history=history_sample,
                                  calendar=calendar_sample,
                                  feature1=feature1_sample,
                                  target=target_sample,
                                  config_summary=["k_best_config_"]
                                  )

            scaled_results_dict[sklearn_estimator.name] = prob_forecast

            rescaled_forecast = target_scaler(x=prob_forecast,
                                              computation_mode=ComputationMode.Transform,
                                              use_inverse_transform=True)

            rescaled_forecast = FunctionModule(correct_shape)(sf=prob_forecast, rf=rescaled_forecast,
                                                              callbacks=[NPYCallback(
                                                                  f'Probabilistic_Forecast_{sklearn_estimator.name}_{tuner_loss_metric.name}')])

            final_results_dict[sklearn_estimator.name] = rescaled_forecast

        for pytorch_estimator in pytorch_estimators:
            deterministic_forecast = pytorch_estimator(x=target_slice,
                                                       calendar=calendar_slice,
                                                       feature1=feature1_slice,
                                                       computation_mode=ComputationMode.Transform)
            scaled_derministic_results[pytorch_estimator.name] = deterministic_forecast

            prob_forecaster = ProbForecastCINN(name=f"{pytorch_estimator.name}_{tuner_loss_metric.name}_prob_forecast",
                                               cinn=cinn_base,
                                               quantiles=cinn_quantiles,
                                               sample_size=cinn_sample_size,
                                               sampling_std=cinn_sampling_std)

            search_space = {
                "sampling_std": hp.uniform("sampling_std", cinn_sampling_lower, cinn_sampling_upper)
            }

            ray_tune_kwargs = {
                "tune_config": tune.TuneConfig(
                    search_alg=HyperOptSearch(space=search_space,
                                              metric="loss", mode="min",
                                              random_state_seed=314),
                    max_concurrent_trials=8,
                    num_samples=-1,  # -1 evaluates infinite configuration samples until time budget is exhausted
                    # time_budget_s is not reset, use run_config
                ),
                "run_config": air.RunConfig(
                    local_dir=f"../../Results/{pipeline_name}/prob_hpo",
                    stop=CombinedStopper(
                        TimeoutStopper(timeout=tuner_timeout),
                        ExperimentPlateauStopper(metric="loss", mode="min", std=0.001, top=10, patience=15 * 40)),
                    # verbose=0
                )
            }

            ray_init_kwargs = {
                # "local_mode": True,  # for debugging
                "log_to_driver": False,
                "logging_level": logging.FATAL
            }

            tuner = RayTuneWrapper(estimator=prob_forecaster, val_split=1,  cv=5, split_method=SplitMethod.FixedOrigin, loss_metric=tuner_loss_metric,
                                   ray_tune_kwargs=ray_tune_kwargs, ray_init_kwargs=ray_init_kwargs, prob_model=True, name=f"Tuner_{pytorch_estimator.name}_{tuner_loss_metric.name}")

            prob_forecast = tuner(input_data=deterministic_forecast,
                                  history=history_sample,
                                  calendar=calendar_sample,
                                  feature1=feature1_sample,
                                  target=target_sample,
                                  config_summary=["k_best_config_"]
                                  )

            scaled_results_dict[pytorch_estimator.name] = prob_forecast

            rescaled_forecast = target_scaler(x=prob_forecast,
                                              computation_mode=ComputationMode.Transform,
                                              use_inverse_transform=True)

            rescaled_forecast = FunctionModule(correct_shape)(sf=prob_forecast, rf=rescaled_forecast,
                                                              callbacks=[NPYCallback(
                                                                  f'Probabilistic_Forecast_{pytorch_estimator.name}_{tuner_loss_metric.name}')])

            final_results_dict[pytorch_estimator.name] = rescaled_forecast

    if number_of_features == 2:
        feature1 = features[0]
        scale_feature1 = feature1_scaler(x=pipeline[feature1])
        feature1_sample = Sampler(24, name="Feature1_Sample")(x=scale_feature1)
        feature1_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature1_sample)
        feature1_slice = ClockShift(lag=24)(x=scale_feature1)
        feature1_slice = Slicer(start=24, end=-24)(x=feature1_slice)
        feature2 = features[1]
        scale_feature2 = feature2_scaler(x=pipeline[feature2])
        feature2_sample = Sampler(24, name="Feature2_Sample")(x=scale_feature2)
        feature2_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature2_sample)
        feature2_slice = ClockShift(lag=24)(x=scale_feature2)
        feature2_slice = Slicer(start=24, end=-24)(x=feature2_slice)

        for sklearn_estimator in sklearn_estimators:
            deterministic_forecast = sklearn_estimator(history=history_sample,
                                                       calendar=calendar_sample,
                                                       feature1=feature1_sample,
                                                       feature2=feature2_sample,
                                                       computation_mode=ComputationMode.Transform)
            scaled_derministic_results[sklearn_estimator.name] = deterministic_forecast

            prob_forecaster = ProbForecastCINN(name=f"{sklearn_estimator.name}_{tuner_loss_metric.name}_prob_forecast",
                                               cinn=cinn_base,
                                               quantiles=cinn_quantiles,
                                               sample_size=cinn_sample_size,
                                               sampling_std=cinn_sampling_std)

            search_space = {
                "sampling_std": hp.uniform("sampling_std", cinn_sampling_lower, cinn_sampling_upper)
            }

            ray_tune_kwargs = {
                "tune_config": tune.TuneConfig(
                    search_alg=HyperOptSearch(space=search_space,
                                              metric="loss", mode="min",
                                              random_state_seed=314),
                    max_concurrent_trials=8,
                    num_samples=-1,  # -1 evaluates infinite configuration samples until time budget is exhausted
                    # time_budget_s is not reset, use run_config
                ),
                "run_config": air.RunConfig(
                    local_dir=f"../../Results/{pipeline_name}/prob_hpo",
                    stop=CombinedStopper(
                        TimeoutStopper(timeout=tuner_timeout),
                        ExperimentPlateauStopper(metric="loss", mode="min", std=0.001, top=10, patience=15 * 40)),
                    # verbose=0
                )
            }

            ray_init_kwargs = {
                #"local_mode": True,  # for debugging
                "log_to_driver": False,
                "logging_level": logging.FATAL
            }

            tuner = RayTuneWrapper(estimator=prob_forecaster, val_split=1,  cv=5, split_method=SplitMethod.FixedOrigin, loss_metric=tuner_loss_metric,
                                   ray_tune_kwargs=ray_tune_kwargs, ray_init_kwargs=ray_init_kwargs, prob_model=True, name=f"Tuner_{sklearn_estimator.name}_{tuner_loss_metric.name}")

            prob_forecast = tuner(input_data=deterministic_forecast,
                                  history=history_sample,
                                  calendar=calendar_sample,
                                  feature1=feature1_sample,
                                  feature2=feature2_sample,
                                  target=target_sample,
                                  config_summary=["k_best_config_"]
                                  )

            scaled_results_dict[sklearn_estimator.name] = prob_forecast

            rescaled_forecast = target_scaler(x=prob_forecast,
                                              computation_mode=ComputationMode.Transform,
                                              use_inverse_transform=True)

            rescaled_forecast = FunctionModule(correct_shape)(sf=prob_forecast, rf=rescaled_forecast,
                                                              callbacks=[NPYCallback(
                                                                  f'Probabilistic_Forecast_{sklearn_estimator.name}_{tuner_loss_metric.name}')])

            final_results_dict[sklearn_estimator.name] = rescaled_forecast

        for pytorch_estimator in pytorch_estimators:
            deterministic_forecast = pytorch_estimator(x=target_slice,
                                                       calendar=calendar_slice,
                                                       feature1=feature1_slice,
                                                       feature2=feature2_slice,
                                                       computation_mode=ComputationMode.Transform)
            scaled_derministic_results[pytorch_estimator.name] = deterministic_forecast

            prob_forecaster = ProbForecastCINN(name=f"{pytorch_estimator.name}_{tuner_loss_metric.name}_prob_forecast",
                                               cinn=cinn_base,
                                               quantiles=cinn_quantiles,
                                               sample_size=cinn_sample_size,
                                               sampling_std=cinn_sampling_std)

            search_space = {
                "sampling_std": hp.uniform("sampling_std", cinn_sampling_lower, cinn_sampling_upper)
            }

            ray_tune_kwargs = {
                "tune_config": tune.TuneConfig(
                    search_alg=HyperOptSearch(space=search_space,
                                              metric="loss", mode="min",
                                              random_state_seed=314),
                    max_concurrent_trials=8,
                    num_samples=-1,  # -1 evaluates infinite configuration samples until time budget is exhausted
                    # time_budget_s is not reset, use run_config
                ),
                "run_config": air.RunConfig(
                    local_dir=f"../../Results/{pipeline_name}/prob_hpo",
                    stop=CombinedStopper(
                        TimeoutStopper(timeout=tuner_timeout),
                        ExperimentPlateauStopper(metric="loss", mode="min", std=0.001, top=10, patience=15 * 40)),
                    # verbose=0
                )
            }

            ray_init_kwargs = {
                # "local_mode": True,  # for debugging
                "log_to_driver": False,
                "logging_level": logging.FATAL
            }

            tuner = RayTuneWrapper(estimator=prob_forecaster, val_split=1,  cv=5, split_method=SplitMethod.FixedOrigin, loss_metric=tuner_loss_metric,
                                   ray_tune_kwargs=ray_tune_kwargs, ray_init_kwargs=ray_init_kwargs, prob_model=True, name=f"Tuner_{pytorch_estimator.name}_{tuner_loss_metric.name}")

            prob_forecast = tuner(input_data=deterministic_forecast,
                                  history=history_sample,
                                  calendar=calendar_sample,
                                  feature1=feature1_sample,
                                  feature2=feature2_sample,
                                  target=target_sample,
                                  config_summary=["k_best_config_"]
                                  )

            scaled_results_dict[pytorch_estimator.name] = prob_forecast

            rescaled_forecast = target_scaler(x=prob_forecast,
                                              computation_mode=ComputationMode.Transform,
                                              use_inverse_transform=True)

            rescaled_forecast = FunctionModule(correct_shape)(sf=prob_forecast, rf=rescaled_forecast,
                                                              callbacks=[NPYCallback(
                                                                  f'Probabilistic_Forecast_{pytorch_estimator.name}_{tuner_loss_metric.name}')])

            final_results_dict[pytorch_estimator.name] = rescaled_forecast

    if number_of_features == 3:
        feature1 = features[0]
        scale_feature1 = feature1_scaler(x=pipeline[feature1])
        feature1_sample = Sampler(24, name="Feature1_Sample")(x=scale_feature1)
        feature1_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature1_sample)
        feature1_slice = ClockShift(lag=24)(x=scale_feature1)
        feature1_slice = Slicer(start=24, end=-24)(x=feature1_slice)
        feature2 = features[1]
        scale_feature2 = feature2_scaler(x=pipeline[feature2])
        feature2_sample = Sampler(24, name="Feature2_Sample")(x=scale_feature2)
        feature2_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature2_sample)
        feature2_slice = ClockShift(lag=24)(x=scale_feature2)
        feature2_slice = Slicer(start=24, end=-24)(x=feature2_slice)
        feature3 = features[2]
        scale_feature3 = feature3_scaler(x=pipeline[feature3])
        feature3_sample = Sampler(24, name="Feature3_Sample")(x=scale_feature3)
        feature3_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature3_sample)
        feature3_slice = ClockShift(lag=24)(x=scale_feature3)
        feature3_slice = Slicer(start=24, end=-24)(x=feature3_slice)

        for sklearn_estimator in sklearn_estimators:
            deterministic_forecast = sklearn_estimator(history=history_sample,
                                                       calendar=calendar_sample,
                                                       feature1=feature1_sample,
                                                       feature2=feature2_sample,
                                                       feature3=feature3_sample,
                                                       computation_mode=ComputationMode.Transform)
            scaled_derministic_results[sklearn_estimator.name] = deterministic_forecast

            prob_forecaster = ProbForecastCINN(name=f"{sklearn_estimator.name}_{tuner_loss_metric.name}_prob_forecast",
                                               cinn=cinn_base,
                                               quantiles=cinn_quantiles,
                                               sample_size=cinn_sample_size,
                                               sampling_std=cinn_sampling_std)

            search_space = {
                "sampling_std": hp.uniform("sampling_std", cinn_sampling_lower, cinn_sampling_upper)
            }

            ray_tune_kwargs = {
                "tune_config": tune.TuneConfig(
                    search_alg=HyperOptSearch(space=search_space,
                                              metric="loss", mode="min",
                                              random_state_seed=314),
                    max_concurrent_trials=8,
                    num_samples=-1,  # -1 evaluates infinite configuration samples until time budget is exhausted
                    # time_budget_s is not reset, use run_config
                ),
                "run_config": air.RunConfig(
                    local_dir=f"../../Results/{pipeline_name}/prob_hpo",
                    stop=CombinedStopper(
                        TimeoutStopper(timeout=tuner_timeout),
                        ExperimentPlateauStopper(metric="loss", mode="min", std=0.001, top=10, patience=15 * 40)),
                    # verbose=0
                )
            }

            ray_init_kwargs = {
                # "local_mode": True,  # for debugging
                "log_to_driver": False,
                "logging_level": logging.FATAL
            }

            tuner = RayTuneWrapper(estimator=prob_forecaster, val_split=1, cv=5, split_method=SplitMethod.FixedOrigin, loss_metric=tuner_loss_metric,
                                   ray_tune_kwargs=ray_tune_kwargs, ray_init_kwargs=ray_init_kwargs, prob_model=True, name=f"Tuner_{sklearn_estimator.name}_{tuner_loss_metric.name}")

            prob_forecast = tuner(input_data=deterministic_forecast,
                                  history=history_sample,
                                  calendar=calendar_sample,
                                  feature1=feature1_sample,
                                  feature2=feature2_sample,
                                  feature3=feature3_sample,
                                  target=target_sample,
                                  config_summary=["k_best_config_"]
                                  )

            scaled_results_dict[sklearn_estimator.name] = prob_forecast

            rescaled_forecast = target_scaler(x=prob_forecast,
                                              computation_mode=ComputationMode.Transform,
                                              use_inverse_transform=True)

            rescaled_forecast = FunctionModule(correct_shape)(sf=prob_forecast, rf=rescaled_forecast,
                                                              callbacks=[NPYCallback(
                                                                  f'Probabilistic_Forecast_{sklearn_estimator.name}_{tuner_loss_metric.name}')])

            final_results_dict[sklearn_estimator.name] = rescaled_forecast

        for pytorch_estimator in pytorch_estimators:
            deterministic_forecast = pytorch_estimator(x=target_slice,
                                                       calendar=calendar_slice,
                                                       feature1=feature1_slice,
                                                       feature2=feature2_slice,
                                                       feature3=feature3_slice,
                                                       computation_mode=ComputationMode.Transform)
            scaled_derministic_results[pytorch_estimator.name] = deterministic_forecast

            prob_forecaster = ProbForecastCINN(name=f"{pytorch_estimator.name}_{tuner_loss_metric.name}_prob_forecast",
                                               cinn=cinn_base,
                                               quantiles=cinn_quantiles,
                                               sample_size=cinn_sample_size,
                                               sampling_std=cinn_sampling_std)

            search_space = {
                "sampling_std": hp.uniform("sampling_std", cinn_sampling_lower, cinn_sampling_upper)
            }

            ray_tune_kwargs = {
                "tune_config": tune.TuneConfig(
                    search_alg=HyperOptSearch(space=search_space,
                                              metric="loss", mode="min",
                                              random_state_seed=314),
                    max_concurrent_trials=8,
                    num_samples=-1,  # -1 evaluates infinite configuration samples until time budget is exhausted
                    # time_budget_s is not reset, use run_config
                ),
                "run_config": air.RunConfig(
                    local_dir=f"../../Results/{pipeline_name}/prob_hpo",
                    stop=CombinedStopper(
                        TimeoutStopper(timeout=tuner_timeout),
                        ExperimentPlateauStopper(metric="loss", mode="min", std=0.001, top=10, patience=15 * 40)),
                    # verbose=0
                )
            }

            ray_init_kwargs = {
                # "local_mode": True,  # for debugging
                "log_to_driver": False,
                "logging_level": logging.FATAL
            }

            tuner = RayTuneWrapper(estimator=prob_forecaster, val_split=1, cv=5, split_method=SplitMethod.FixedOrigin, loss_metric=tuner_loss_metric,
                                   ray_tune_kwargs=ray_tune_kwargs, ray_init_kwargs=ray_init_kwargs, prob_model=True, name=f"Tuner_{pytorch_estimator.name}_{tuner_loss_metric.name}")

            prob_forecast = tuner(input_data=deterministic_forecast,
                                  history=history_sample,
                                  calendar=calendar_sample,
                                  feature1=feature1_sample,
                                  feature2=feature2_sample,
                                  feature3=feature3_sample,
                                  target=target_sample,
                                  config_summary=["k_best_config_"]
                                  )

            scaled_results_dict[pytorch_estimator.name] = prob_forecast

            rescaled_forecast = target_scaler(x=prob_forecast,
                                              computation_mode=ComputationMode.Transform,
                                              use_inverse_transform=True)

            rescaled_forecast = FunctionModule(correct_shape)(sf=prob_forecast, rf=rescaled_forecast,
                                                              callbacks=[NPYCallback(
                                                                  f'Probabilistic_Forecast_{pytorch_estimator.name}_{tuner_loss_metric.name}')])

            final_results_dict[pytorch_estimator.name] = rescaled_forecast

    if number_of_features == 4:
        feature1 = features[0]
        scale_feature1 = feature1_scaler(x=pipeline[feature1])
        feature1_sample = Sampler(24, name="Feature1_Sample")(x=scale_feature1)
        feature1_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature1_sample)
        feature1_slice = ClockShift(lag=24)(x=scale_feature1)
        feature1_slice = Slicer(start=24, end=-24)(x=feature1_slice)
        feature2 = features[1]
        scale_feature2 = feature2_scaler(x=pipeline[feature2])
        feature2_sample = Sampler(24, name="Feature2_Sample")(x=scale_feature2)
        feature2_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature2_sample)
        feature2_slice = ClockShift(lag=24)(x=scale_feature2)
        feature2_slice = Slicer(start=24, end=-24)(x=feature2_slice)
        feature3 = features[2]
        scale_feature3 = feature3_scaler(x=pipeline[feature3])
        feature3_sample = Sampler(24, name="Feature3_Sample")(x=scale_feature3)
        feature3_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature3_sample)
        feature3_slice = ClockShift(lag=24)(x=scale_feature3)
        feature3_slice = Slicer(start=24, end=-24)(x=feature3_slice)
        feature4 = features[3]
        scale_feature4 = feature4_scaler(x=pipeline[feature4])
        feature4_sample = Sampler(24, name="Feature4_Sample")(x=scale_feature4)
        feature4_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature4_sample)
        feature4_slice = ClockShift(lag=24)(x=scale_feature4)
        feature4_slice = Slicer(start=24, end=-24)(x=feature4_slice)

        for sklearn_estimator in sklearn_estimators:
            deterministic_forecast = sklearn_estimator(history=history_sample,
                                                       calendar=calendar_sample,
                                                       feature1=feature1_sample,
                                                       feature2=feature2_sample,
                                                       feature3=feature3_sample,
                                                       feature4=feature4_sample,
                                                       computation_mode=ComputationMode.Transform)
            scaled_derministic_results[sklearn_estimator.name] = deterministic_forecast

            prob_forecaster = ProbForecastCINN(name=f"{sklearn_estimator.name}_{tuner_loss_metric.name}_prob_forecast",
                                               cinn=cinn_base,
                                               quantiles=cinn_quantiles,
                                               sample_size=cinn_sample_size,
                                               sampling_std=cinn_sampling_std)

            search_space = {
                "sampling_std": hp.uniform("sampling_std", cinn_sampling_lower, cinn_sampling_upper)
            }

            ray_tune_kwargs = {
                "tune_config": tune.TuneConfig(
                    search_alg=HyperOptSearch(space=search_space,
                                              metric="loss", mode="min",
                                              random_state_seed=314),
                    max_concurrent_trials=8,
                    num_samples=-1,  # -1 evaluates infinite configuration samples until time budget is exhausted
                    # time_budget_s is not reset, use run_config
                ),
                "run_config": air.RunConfig(
                    local_dir=f"../../Results/{pipeline_name}/prob_hpo",
                    stop=CombinedStopper(
                        TimeoutStopper(timeout=tuner_timeout),
                        ExperimentPlateauStopper(metric="loss", mode="min", std=0.001, top=10, patience=15 * 40)),
                    # verbose=0
                )
            }

            ray_init_kwargs = {
                #"local_mode": True,  # for debugging
                "log_to_driver": False,
                "logging_level": logging.FATAL
            }

            tuner = RayTuneWrapper(estimator=prob_forecaster, val_split=1, cv=5, split_method=SplitMethod.FixedOrigin, loss_metric=tuner_loss_metric,
                                   ray_tune_kwargs=ray_tune_kwargs, ray_init_kwargs=ray_init_kwargs, prob_model=True, name=f"Tuner_{sklearn_estimator.name}_{tuner_loss_metric.name}")

            prob_forecast = tuner(input_data=deterministic_forecast,
                                  history=history_sample,
                                  calendar=calendar_sample,
                                  feature1=feature1_sample,
                                  feature2=feature2_sample,
                                  feature3=feature3_sample,
                                  feature4=feature4_sample,
                                  target=target_sample,
                                  config_summary=["k_best_config_"]
                                  )

            scaled_results_dict[sklearn_estimator.name] = prob_forecast

            rescaled_forecast = target_scaler(x=prob_forecast,
                                              computation_mode=ComputationMode.Transform,
                                              use_inverse_transform=True)

            rescaled_forecast = FunctionModule(correct_shape)(sf=prob_forecast, rf=rescaled_forecast,
                                                              callbacks=[NPYCallback(
                                                                  f'Probabilistic_Forecast_{sklearn_estimator.name}_{tuner_loss_metric.name}')])

            final_results_dict[sklearn_estimator.name] = rescaled_forecast

        for pytorch_estimator in pytorch_estimators:
            deterministic_forecast = pytorch_estimator(x=target_slice,
                                                       calendar=calendar_slice,
                                                       feature1=feature1_slice,
                                                       feature2=feature2_slice,
                                                       feature3=feature3_slice,
                                                       feature4=feature4_slice,
                                                       computation_mode=ComputationMode.Transform)
            scaled_derministic_results[pytorch_estimator.name] = deterministic_forecast

            prob_forecaster = ProbForecastCINN(name=f"{pytorch_estimator.name}_{tuner_loss_metric.name}_prob_forecast",
                                               cinn=cinn_base,
                                               quantiles=cinn_quantiles,
                                               sample_size=cinn_sample_size,
                                               sampling_std=cinn_sampling_std)

            search_space = {
                "sampling_std": hp.uniform("sampling_std", cinn_sampling_lower, cinn_sampling_upper)
            }

            ray_tune_kwargs = {
                "tune_config": tune.TuneConfig(
                    search_alg=HyperOptSearch(space=search_space,
                                              metric="loss", mode="min",
                                              random_state_seed=314),
                    max_concurrent_trials=8,
                    num_samples=-1,  # -1 evaluates infinite configuration samples until time budget is exhausted
                    # time_budget_s is not reset, use run_config
                ),
                "run_config": air.RunConfig(
                    local_dir=f"../../Results/{pipeline_name}/prob_hpo",
                    stop=CombinedStopper(
                        TimeoutStopper(timeout=tuner_timeout),
                        ExperimentPlateauStopper(metric="loss", mode="min", std=0.001, top=10, patience=15 * 40)),
                    # verbose=0
                )
            }

            ray_init_kwargs = {
                # "local_mode": True,  # for debugging
                "log_to_driver": False,
                "logging_level": logging.FATAL
            }

            tuner = RayTuneWrapper(estimator=prob_forecaster, val_split=1, cv=5, split_method=SplitMethod.FixedOrigin, loss_metric=tuner_loss_metric,
                                   ray_tune_kwargs=ray_tune_kwargs, ray_init_kwargs=ray_init_kwargs, prob_model=True, name=f"Tuner_{pytorch_estimator.name}_{tuner_loss_metric.name}")

            prob_forecast = tuner(input_data=deterministic_forecast,
                                  history=history_sample,
                                  calendar=calendar_sample,
                                  feature1=feature1_sample,
                                  feature2=feature2_sample,
                                  feature3=feature3_sample,
                                  feature4=feature4_sample,
                                  target=target_sample,
                                  config_summary=["k_best_config_"]
                                  )

            scaled_results_dict[pytorch_estimator.name] = prob_forecast

            rescaled_forecast = target_scaler(x=prob_forecast,
                                              computation_mode=ComputationMode.Transform,
                                              use_inverse_transform=True)

            rescaled_forecast = FunctionModule(correct_shape)(sf=prob_forecast, rf=rescaled_forecast,
                                                              callbacks=[NPYCallback(
                                                                  f'Probabilistic_Forecast_{pytorch_estimator.name}_{tuner_loss_metric.name}')])

            final_results_dict[pytorch_estimator.name] = rescaled_forecast

    if number_of_features >= 5:
        raise ValueError("Error: Features not considered") from Exception(
            "Currently a maximum of four features are supported")

    CRPS(name=f"CRPS_{tuner_loss_metric.name}")(**scaled_results_dict, y=target_sample)
    Pinball_Loss(name=f"Pinball_{tuner_loss_metric.name}")(**scaled_results_dict, y=target_sample)
    Coverage_Rate(name=f"Coverage_{tuner_loss_metric.name}")(**scaled_results_dict, y=target_sample)
    MAE(name=f"MAE_{tuner_loss_metric.name}")(**scaled_derministic_results, y=target_sample),
    RMSE(name=f"RMSE_{tuner_loss_metric.name}")(**scaled_derministic_results,y=target_sample)

    return pipeline, final_results_dict


def correct_shape(sf, rf):
    new_array = numpy_to_xarray(rf.values.reshape(sf.shape), sf)
    new_array = new_array.rename({'dim_0': 'horizon', 'dim_1': 'quantiles'})
    return new_array


def flatten(d):
    result = {}
    if isinstance(d, dict):
        for o_key, o_value in d.items():
            result.update({o_key + "_" + i_key: i_value for i_key, i_value in flatten(o_value).items()})
        return result
    else:
        return {"": d}

def get_keras_model(features):

    number_of_features = len(features)

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

    elif number_of_features == 1:
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

    elif number_of_features == 2:
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

    elif number_of_features == 3:
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

    elif number_of_features == 4:
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
    else:
        model = 0

    return model
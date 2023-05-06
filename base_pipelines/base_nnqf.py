import numpy as np
from pywatts.core.computation_mode import ComputationMode
from pywatts.core.pipeline import Pipeline
from pywatts.utils._xarray_time_series_utils import numpy_to_xarray
from pywatts.callbacks.npy_callback import NPYCallback
from modules.pinball import Pinball_Loss
from pywatts.modules import ClockShift, SKLearnWrapper, Sampler, FunctionModule, Slicer

from sklearn.preprocessing import StandardScaler

from modules.crps import CRPS
from modules.CR import Coverage_Rate
from modules.nnqf import NNQF


def train_forecast_pipeline(pipeline_name, target, sklearn_estimators, calendar_extraction, features):
    pipeline = Pipeline(f"../../Results/{pipeline_name}/train_nnqf", name="train_nnqf")

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

    # Deal with features
    number_of_features = len(features)
    feature1_scaler = None
    feature2_scaler = None
    feature3_scaler = None
    feature4_scaler = None

    if number_of_features == 0:

        nnqf = NNQF()(y=scale_target,
                      calendar=calendar,
                      history=target_history)
        # Sample target
        nnqf_target_sample = Sampler(24, name=f"Target_Sample")(x=nnqf)
        nnqf_target_sample = Slicer(start=2 * 24, end=-2 * 24)(x=nnqf_target_sample)
        target_sample = FunctionModule(prep_target)(x=nnqf_target_sample)

        for sklearn_estimator in sklearn_estimators:
            sklearn_estimator(history=history_sample,
                              calendar=calendar_sample,
                              target=target_sample)

    if number_of_features == 1:
        feature1 = features[0]
        feature1_scaler = SKLearnWrapper(module=StandardScaler(), name="feature1_scaler")
        scale_feature1 = feature1_scaler(x=pipeline[feature1])
        feature1_sample = Sampler(24, name="Feature1_Sample")(x=scale_feature1)
        feature1_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature1_sample)

        nnqf = NNQF()(y=scale_target,
                      calendar=calendar,
                      history=target_history,
                      feature1=scale_feature1)
        # Sample target
        nnqf_target_sample = Sampler(24, name=f"Target_Sample")(x=nnqf)
        nnqf_target_sample = Slicer(start=2 * 24, end=-2 * 24)(x=nnqf_target_sample)
        target_sample = FunctionModule(prep_target)(x=nnqf_target_sample)

        for sklearn_estimator in sklearn_estimators:
            sklearn_estimator(history=history_sample,
                              calendar=calendar_sample,
                              feature1=feature1_sample,
                              target=target_sample)

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

        nnqf = NNQF()(y=scale_target,
                      calendar=calendar,
                      history=target_history,
                      feature1=scale_feature1,
                      feature2=scale_feature2)
        # Sample target
        nnqf_target_sample = Sampler(24, name=f"Target_Sample")(x=nnqf)
        nnqf_target_sample = Slicer(start=2 * 24, end=-2 * 24)(x=nnqf_target_sample)
        target_sample = FunctionModule(prep_target)(x=nnqf_target_sample)

        for sklearn_estimator in sklearn_estimators:
            sklearn_estimator(history=history_sample,
                              calendar=calendar_sample,
                              feature1=feature1_sample,
                              feature2=feature2_sample,
                              target=target_sample)

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

        nnqf = NNQF()(y=scale_target,
                      calendar=calendar,
                      history=target_history,
                      feature1=scale_feature1,
                      feature2=scale_feature2,
                      feature3=scale_feature3)

        # Sample target
        nnqf_target_sample = Sampler(24, name=f"Target_Sample")(x=nnqf)
        nnqf_target_sample = Slicer(start=2 * 24, end=-2 * 24)(x=nnqf_target_sample)
        target_sample = FunctionModule(prep_target)(x=nnqf_target_sample)

        for sklearn_estimator in sklearn_estimators:
            sklearn_estimator(history=history_sample,
                              calendar=calendar_sample,
                              feature1=feature1_sample,
                              feature2=feature2_sample,
                              feature3=feature3_sample,
                              target=target_sample)

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

        nnqf = NNQF()(y=scale_target,
                      calendar=calendar,
                      history=target_history,
                      feature1=scale_feature1,
                      feature2=scale_feature2,
                      feature3=scale_feature3,
                      feature4=scale_feature4)

        # Sample target
        nnqf_target_sample = Sampler(24, name=f"Target_Sample")(x=nnqf)
        nnqf_target_sample = Slicer(start=2 * 24, end=-2 * 24)(x=nnqf_target_sample)
        target_sample = FunctionModule(prep_target)(x=nnqf_target_sample)

        for sklearn_estimator in sklearn_estimators:
            sklearn_estimator(history=history_sample,
                              calendar=calendar_sample,
                              feature1=feature1_sample,
                              feature2=feature2_sample,
                              feature3=feature3_sample,
                              feature4=feature4_sample,
                              target=target_sample)

    if number_of_features >= 5:
        raise ValueError("Error: Features not considered") from Exception(
            "Currently a maximum of four features are supported")

    return pipeline, target_scaler, feature1_scaler, feature2_scaler, feature3_scaler, feature4_scaler


def prep_target(x):
    return numpy_to_xarray(x.values.swapaxes(1, 2), x)


def get_prob_forecast_pipeline(pipeline_name, target, calendar_extraction, target_scaler, sklearn_estimators, features,
                               feature1_scaler, feature2_scaler, feature3_scaler, feature4_scaler):
    pipeline = Pipeline(f"../../Results/{pipeline_name}/evaluate_nnqf", name=f"evaluate_nnqf")

    # Extract calendar features
    calendar = calendar_extraction(x=pipeline[target])
    calendar_sample = Sampler(24, name="CalendarSample")(x=calendar)
    calendar_sample = Slicer(start=2 * 24, end=-2 * 24)(x=calendar_sample)

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

    # Create Historical Values and Sample
    target_history = ClockShift(lag=25, name="History_Target")(x=scale_target)
    history_sample = Sampler(24, name="History_Sample")(x=target_history)
    history_sample = Slicer(start=2 * 24, end=-2 * 24)(x=history_sample)

    # Deal with features
    number_of_features = len(features)

    # Save Results
    scaled_results_dict = {}
    final_results_dict = {}

    if number_of_features == 0:
        for sklearn_estimator in sklearn_estimators:
            forecast = sklearn_estimator(history=history_sample,
                                         calendar=calendar_sample,
                                         computation_mode=ComputationMode.Transform)
            prob_forecast = FunctionModule(prep_forecast)(x=forecast)

            scaled_results_dict[sklearn_estimator.name] = prob_forecast

            rescaled_forecast = target_scaler(x=prob_forecast,
                                              computation_mode=ComputationMode.Transform,
                                              use_inverse_transform=True)

            rescaled_forecast = FunctionModule(correct_shape)(sf=prob_forecast, rf=rescaled_forecast,
                                                              callbacks=[NPYCallback(
                                                                  f'NNQF_Forecast_{sklearn_estimator.name}')])

            final_results_dict[sklearn_estimator.name] = rescaled_forecast

    if number_of_features == 1:
        feature1 = features[0]
        scale_feature1 = feature1_scaler(x=pipeline[feature1])
        feature1_sample = Sampler(24, name="Feature1_Sample")(x=scale_feature1)
        feature1_sample = Slicer(start=2 * 24, end=-2 * 24)(x=feature1_sample)
        feature1_slice = ClockShift(lag=24)(x=scale_feature1)
        feature1_slice = Slicer(start=24, end=-24)(x=feature1_slice)

        for sklearn_estimator in sklearn_estimators:
            forecast = sklearn_estimator(history=history_sample,
                                         calendar=calendar_sample,
                                         feature1=feature1_sample,
                                         computation_mode=ComputationMode.Transform)
            prob_forecast = FunctionModule(prep_forecast)(x=forecast)

            scaled_results_dict[sklearn_estimator.name] = prob_forecast

            rescaled_forecast = target_scaler(x=prob_forecast,
                                              computation_mode=ComputationMode.Transform,
                                              use_inverse_transform=True)

            rescaled_forecast = FunctionModule(correct_shape)(sf=prob_forecast, rf=rescaled_forecast,
                                                              callbacks=[NPYCallback(
                                                                  f'NNQF_Forecast_{sklearn_estimator.name}')])

            final_results_dict[sklearn_estimator.name] = rescaled_forecast

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
            forecast = sklearn_estimator(history=history_sample,
                                         calendar=calendar_sample,
                                         feature1=feature1_sample,
                                         feature2=feature2_sample,
                                         computation_mode=ComputationMode.Transform)
            prob_forecast = FunctionModule(prep_forecast)(x=forecast)

            scaled_results_dict[sklearn_estimator.name] = prob_forecast

            rescaled_forecast = target_scaler(x=prob_forecast,
                                              computation_mode=ComputationMode.Transform,
                                              use_inverse_transform=True)

            rescaled_forecast = FunctionModule(correct_shape)(sf=prob_forecast, rf=rescaled_forecast,
                                                              callbacks=[NPYCallback(
                                                                  f'NNQF_Forecast_{sklearn_estimator.name}')])

            final_results_dict[sklearn_estimator.name] = rescaled_forecast

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
            forecast = sklearn_estimator(history=history_sample,
                                         calendar=calendar_sample,
                                         feature1=feature1_sample,
                                         feature2=feature2_sample,
                                         feature3=feature3_sample,
                                         computation_mode=ComputationMode.Transform)
            prob_forecast = FunctionModule(prep_forecast)(x=forecast)

            scaled_results_dict[sklearn_estimator.name] = prob_forecast

            rescaled_forecast = target_scaler(x=prob_forecast,
                                              computation_mode=ComputationMode.Transform,
                                              use_inverse_transform=True)

            rescaled_forecast = FunctionModule(correct_shape)(sf=prob_forecast, rf=rescaled_forecast,
                                                              callbacks=[NPYCallback(
                                                                  f'NNQF_Forecast_{sklearn_estimator.name}')])

            final_results_dict[sklearn_estimator.name] = rescaled_forecast

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
            forecast = sklearn_estimator(history=history_sample,
                                         calendar=calendar_sample,
                                         feature1=feature1_sample,
                                         feature2=feature2_sample,
                                         feature3=feature3_sample,
                                         feature4=feature4_sample,
                                         computation_mode=ComputationMode.Transform)
            prob_forecast = FunctionModule(prep_forecast)(x=forecast)

            scaled_results_dict[sklearn_estimator.name] = prob_forecast

            rescaled_forecast = target_scaler(x=prob_forecast,
                                              computation_mode=ComputationMode.Transform,
                                              use_inverse_transform=True)

            rescaled_forecast = FunctionModule(correct_shape)(sf=prob_forecast, rf=rescaled_forecast,
                                                              callbacks=[NPYCallback(
                                                                  f'NNQF_Forecast_{sklearn_estimator.name}')])

            final_results_dict[sklearn_estimator.name] = rescaled_forecast

    if number_of_features >= 5:
        raise ValueError("Error: Features not considered") from Exception(
            "Currently a maximum of four features are supported")

    CRPS(name=f"CRPS")(**scaled_results_dict, y=target_sample)
    Pinball_Loss(name=f"Pinball")(**scaled_results_dict, y=target_sample)
    Coverage_Rate(name=f"Coverage")(**scaled_results_dict, y=target_sample)

    return pipeline, final_results_dict


def prep_forecast(x):
    x = numpy_to_xarray(x.values.reshape(len(x), 99, 24).swapaxes(1, 2), x)
    x = x.rename({'dim_0': 'horizon', 'dim_1': 'quantiles'})
    final_array = x.assign_coords(coords={"quantiles": np.arange(1,100,1)})
    return final_array


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

import pandas as pd
import os
from pywatts.modules import CalendarExtraction, CalendarFeature, SKLearnWrapper, KerasWrapper
from pywatts.core.summary_formatter import SummaryJSON

from modules.hyperparameter_optimization import LossMetric
from modules.inn import INNWrapper
from modules.pytorch_forecasting_determ_wrapper import PyTorchForecastingDeterministicWrapper

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pytorch_forecasting.models import NHiTS, TemporalFusionTransformer

from base_pipelines.base_cinn_autoHPO import train_forecast_pipeline, train_cinn_pipeline, get_prob_forecast_pipeline, flatten, \
    get_keras_model

PIPELINE_NAME = "solar_cinn"
TARGET = "POWER"
FEATURES = ["SSRD", "TCC"]

QUANTILES = [50, 1, 99, 5, 95, 15, 85, 25, 75, 10, 90, 20, 80, 30, 70, 40, 60]
INN_STD_DEFAULT = 0.35
INN_STD_LOWER = 0.1
INN_STD_UPPER = 2
INN_EPOCHS = 100
INN_SAMPLE_SIZE = 100
TUNER_METRICS = [LossMetric.CRPS, LossMetric.PINBALL, LossMetric.CRE, LossMetric.CAL,
                 LossMetric.WPI, LossMetric.WPI_Weighted]
TUNER_TIMEOUT = 2 * 60
NUMER_OF_RUNS = 3


def prepare_data():
    data = pd.read_csv("../../data/solar.csv")
    data.index = pd.date_range(start="4/1/2012 01:00:00", freq="1H", periods=len(data))
    data.index.name = 'time'
    data = data.drop(columns=["Unnamed: 0"])

    train = data.iloc[:11034, :]
    val = data.iloc[11034:15763, :]
    test = data.iloc[15763:19704, :]

    return data, train, val, test


if __name__ == "__main__":
    # Split Data
    data, train, val, test = prepare_data()

    df_eval = pd.DataFrame()
    df_param = pd.DataFrame()

    for i in range(NUMER_OF_RUNS):

        neural_network = get_keras_model(FEATURES)

        # Build Estimators
        sklearn_estimators = [SKLearnWrapper(module=LinearRegression(), name="Linear_regression"),
                              SKLearnWrapper(module=RandomForestRegressor(), name="RF_Regression"),
                              SKLearnWrapper(module=MLPRegressor(), name="MLP_Regression"),
                              KerasWrapper(neural_network, fit_kwargs={"batch_size": 100, "epochs": 100},
                                           compile_kwargs={"loss": "mse", "optimizer": "Adam", "metrics": ["mse"]},
                                           name="NN_Regression"),
                              SKLearnWrapper(module=XGBRegressor(), name="XGB_Regression")]

        pytorch_estimators = [PyTorchForecastingDeterministicWrapper(NHiTS, name="N-HITS"),
                              PyTorchForecastingDeterministicWrapper(TemporalFusionTransformer, name="TFT")]


        # Calendar Information
        calendar_extraction = CalendarExtraction(continent="Europe",
                                                 country="Germany",
                                                 features=[CalendarFeature.workday, CalendarFeature.hour_cos,
                                                           CalendarFeature.hour_sine, CalendarFeature.month_sine,
                                                           CalendarFeature.month_cos])

        # Build deterministic forecasting pipeline and train
        deterministic_forecasting_pipeline, target_scaler, feature1_scaler, feature2_scaler, feature3_scaler, feature4_scaler = train_forecast_pipeline(
            pipeline_name=PIPELINE_NAME,
            target=TARGET,
            sklearn_estimators=sklearn_estimators,
            pytorch_estimators=pytorch_estimators,
            calendar_extraction=calendar_extraction,
            features=FEATURES)
        deterministic_forecasting_pipeline.train(data=train, summary=True, summary_formatter=SummaryJSON())

        # Create cINN
        cinn_model = INNWrapper(name="INN", quantiles=QUANTILES, sample_size=INN_SAMPLE_SIZE)

        # Build cINN Pipeline and train
        cinn_pipeline, cinn = train_cinn_pipeline(pipeline_name=PIPELINE_NAME,
                                                  target=TARGET,
                                                  calendar_extraction=calendar_extraction,
                                                  cinn=cinn_model,
                                                  cinn_epochs=INN_EPOCHS,
                                                  features=FEATURES)

        cinn_pipeline.train(data=train, summary=True, summary_formatter=SummaryJSON())

        for tuner_metric in TUNER_METRICS:
            prob_forecast_pipeline, prob_forecast = get_prob_forecast_pipeline(pipeline_name=PIPELINE_NAME,
                                                                               target=TARGET,
                                                                               calendar_extraction=calendar_extraction,
                                                                               target_scaler=target_scaler,
                                                                               sklearn_estimators=sklearn_estimators,
                                                                               pytorch_estimators=pytorch_estimators,
                                                                               cinn_base=cinn,
                                                                               cinn_quantiles=QUANTILES,
                                                                               cinn_sample_size=INN_SAMPLE_SIZE,
                                                                               cinn_sampling_std=INN_STD_DEFAULT,
                                                                               cinn_sampling_lower=INN_STD_LOWER,
                                                                               cinn_sampling_upper=INN_STD_UPPER,
                                                                               tuner_loss_metric=tuner_metric,
                                                                               tuner_timeout=TUNER_TIMEOUT,
                                                                               features=FEATURES,
                                                                               feature1_scaler=feature1_scaler,
                                                                               feature2_scaler=feature2_scaler,
                                                                               feature3_scaler=feature3_scaler,
                                                                               feature4_scaler=feature4_scaler)

            prob_forecast_pipeline.train(data=val, summary=True, summary_formatter=SummaryJSON())

            result, summary = prob_forecast_pipeline.test(data=test, summary=True, summary_formatter=SummaryJSON())

            df_eval = df_eval.append(flatten(summary["Summary"]), ignore_index=True)
            df_param = df_param.append(flatten(summary["FitConfiguration"]), ignore_index=True)

            print("###################################################################################################"
                  "#####################################################################################"
                  "############################################################################################")
            print(f"############################################################################################ "
                  f"For Run {i+1}, finished {tuner_metric.name} ##################################################"
                  f"##########################################")
            print("###################################################################################################"
                  "#####################################################################################"
                  "############################################################################################")

        print("**************************************************************************************************"
              "****************************************************************************************************"
              "****************************************************************** *")
        print(f"************************************************************************************** "
              f"Finished Run {i+1} ***********************************************************************"
              f"***************************")
        print("**************************************************************************************************"
              "****************************************************************************************************"
              "****************************************************************** *")

    outdir = "../../Summaries"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    df_eval.to_csv(f"{outdir}/evaluation_{PIPELINE_NAME}.csv")
    df_param.to_csv(f"{outdir}/parameters_{PIPELINE_NAME}.csv")

    print("Finished all runs!")
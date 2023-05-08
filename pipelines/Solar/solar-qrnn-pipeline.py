import pandas as pd
import os
from pywatts.modules import CalendarExtraction, CalendarFeature
from pywatts.core.summary_formatter import SummaryJSON

from modules.QRNetwork import PLNN

from base_pipelines.base_qrnn import get_prob_forecast_pipeline, train_forecast_pipeline, flatten

PIPELINE_NAME = "solar_qrnn"
TARGET = "POWER"
FEATURES = ["SSRD", "TCC"]

QUANTILES = [50, 1, 99, 5, 95, 15, 85, 25, 75, 10, 90, 20, 80, 30, 70, 40, 60]
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
    quantiles = [x / 100 for x in QUANTILES]

    for i in range(NUMER_OF_RUNS):

        # Build Estimators
        qrnn = PLNN(q_values=quantiles, number_of_features=len(FEATURES))

        # Calendar Information
        calendar_extraction = CalendarExtraction(continent="Europe",
                                                 country="Germany",
                                                 features=[CalendarFeature.workday, CalendarFeature.hour_cos,
                                                           CalendarFeature.hour_sine, CalendarFeature.month_sine,
                                                           CalendarFeature.month_cos])

        # Build deterministic forecasting pipeline and train
        forecasting_pipeline, target_scaler, feature1_scaler, feature2_scaler, feature3_scaler, feature4_scaler = train_forecast_pipeline(
            pipeline_name=PIPELINE_NAME,
            target=TARGET,
            qrnn=qrnn,
            calendar_extraction=calendar_extraction,
            features=FEATURES)
        forecasting_pipeline.train(data=train, summary=True, summary_formatter=SummaryJSON())

        prob_forecast_pipeline, prob_forecast = get_prob_forecast_pipeline(pipeline_name=PIPELINE_NAME,
                                                                               target=TARGET,
                                                                               calendar_extraction=calendar_extraction,
                                                                               target_scaler=target_scaler,
                                                                               qrnn=qrnn,
                                                                               features=FEATURES,
                                                                               feature1_scaler=feature1_scaler,
                                                                               feature2_scaler=feature2_scaler,
                                                                               feature3_scaler=feature3_scaler,
                                                                               feature4_scaler=feature4_scaler)

        result, summary = prob_forecast_pipeline.test(data=test, summary=True, summary_formatter=SummaryJSON())

        df_eval = df_eval.append(flatten(summary["Summary"]), ignore_index=True)

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

    print("Finished all runs!")

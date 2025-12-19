from darts import TimeSeries
from tqdm.auto import tqdm
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Optional
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MaxAbsScaler
import warnings
warnings.filterwarnings("ignore")

warnings.simplefilter("ignore", UserWarning)



def load_m3(HORIZON) -> tuple[list[TimeSeries], list[TimeSeries]]:
    print("building M3 TimeSeries...")

    # Read DataFrame
    df_m3 = pd.read_excel("m3_dataset.xls", "M3Month")

    # Build TimeSeries
    m3_series = []
    for row in tqdm(df_m3.iterrows()):
        s = row[1]
        start_year = int(s["Starting Year"])
        start_month = int(s["Starting Month"])
        values_series = s[6:].dropna()
        if start_month == 0:
            continue

        start_date = datetime(year=start_year, month=start_month, day=1)
        time_axis = pd.date_range(start_date, periods=len(values_series), freq="M")
        series = TimeSeries.from_times_and_values(
            time_axis, values_series.values
        ).astype(np.float32)
        m3_series.append(series)

    print(f"\nThere are {len(m3_series)} monthly series in the M3 dataset")

    # Split train/test
    print("splitting train/test...")
    m3_train = [s[:-HORIZON] for s in m3_series]
    m3_test = [s[-HORIZON:] for s in m3_series]

    # Scale so that the largest value is 1
    print("scaling...")
    scaler_m3 = Scaler(scaler=MaxAbsScaler())
    m3_train_scaled: list[TimeSeries] = scaler_m3.fit_transform(m3_train)
    m3_test_scaled: list[TimeSeries] = scaler_m3.transform(m3_test)

    print(
        f"done. There are {len(m3_train_scaled)} series, with average training length {np.mean([len(s) for s in m3_train_scaled])}"  # noqa: E501
    )
    return m3_train_scaled, m3_test_scaled


def load_air(HORIZON) -> tuple[list[TimeSeries], list[TimeSeries]]:
    # download csv file
    df = pd.read_csv("carrier_passengers.csv")
    # extract relevant columns
    df = df[["data_dte", "carrier", "Total"]]
    # aggregate per carrier and date
    df = pd.DataFrame(df.groupby(["carrier", "data_dte"]).sum())
    # move indexes to columns
    print(df)
    df = df.reset_index()

    # group bt carrier, specify time index and target variable
    all_air_series = TimeSeries.from_group_dataframe(
        df, group_cols="carrier", time_col="data_dte", value_cols="Total", freq="MS"
    )

    # Split train/test
    print("splitting train/test...")
    air_train = []
    air_test = []
    for series in all_air_series:
        # remove the end of the series
        series = series[: pd.Timestamp("2019-12-31")]
        # convert to proper type
        series = series.astype(np.float32)
        # extract longest contiguous slice
        try:
            series = series.longest_contiguous_slice()
        except Exception:
            continue
        # remove static covariates
        series = series.with_static_covariates(None)
        # remove short series
        if len(series) >= 36 + HORIZON:
            air_train.append(series[:-HORIZON])
            air_test.append(series[-HORIZON:])

    # Scale so that the largest value is 1
    print("scaling series...")
    scaler_air = Scaler(scaler=MaxAbsScaler())
    air_train_scaled: list[TimeSeries] = scaler_air.fit_transform(air_train)
    air_test_scaled: list[TimeSeries] = scaler_air.transform(air_test)

    print(
        f"done. There are {len(air_train_scaled)} series, with average training length {np.mean([len(s) for s in air_train_scaled])}"  # noqa: E501
    )
    return air_train_scaled, air_test_scaled


def load_m4(HORIZON,
    max_number_series: Optional[int] = None,
) -> tuple[list[TimeSeries], list[TimeSeries]]:
    """
    Due to the size of the dataset, this function takes approximately 10 minutes.

    Use the `max_number_series` parameter to reduce the computation time if necessary
    """
    # Read data dataFrame
    df_m4 = pd.read_csv("m4_monthly.csv")
    if max_number_series is not None:
        df_m4 = df_m4[:max_number_series]
    # Read metadata dataframe
    df_meta = pd.read_csv("m4_metadata.csv")
    df_meta = df_meta.loc[df_meta.SP == "Monthly"]

    # Build TimeSeries
    m4_train = []
    m4_test = []
    for row in tqdm(df_m4.iterrows(), total=len(df_m4)):
        s = row[1]
        values_series = s[1:].dropna()
        start_date = pd.Timestamp(
            df_meta.loc[df_meta["M4id"] == "M1", "StartingDate"].values[0]
        )
        time_axis = pd.date_range(start_date, periods=len(values_series), freq="M")
        series = TimeSeries.from_times_and_values(
            time_axis, values_series.values
        ).astype(np.float32)
        # remove series with less than 48 training samples
        if len(series) > 48 + HORIZON:
            # Split train/test
            m4_train.append(series[:-HORIZON])
            m4_test.append(series[-HORIZON:])

    print(f"\nThere are {len(m4_train)} monthly series in the M3 dataset")

    # Scale so that the largest value is 1
    print("scaling...")
    scaler_m4 = Scaler(scaler=MaxAbsScaler())
    m4_train_scaled: list[TimeSeries] = scaler_m4.fit_transform(m4_train)
    m4_test_scaled: list[TimeSeries] = scaler_m4.transform(m4_test)

    print(
        f"done. There are {len(m4_train_scaled)} series, with average training length {np.mean([len(s) for s in m4_train_scaled])}"  # noqa: E501
    )
    return m4_train_scaled, m4_test_scaled
import pickle
import datetime
import pandas as pd
import numpy as np


def load_data(fpath):
    df = pd.read_parquet(fpath)
    df.target = series2date(df.target)
    return df


def series2date(series):
    """
    Check dtype of pd.Series, convert to datetime.date
    series: pandas Series with dates in format "yyyy-mm-dd"
    """
    """if isinstance(series[0], str):
        series = series.apply(datetime.datetime.strptime, args=['%Y-%m-%d']).dt.date
    elif isinstance(series[0], datetime.datetime):
        series = series.apply(pd.Timestamp.date)"""
    #if isinstance(series[0], str) or isinstance(series[0], datetime.datetime):
    series = series.astype(np.datetime64)
    return series


def load_results(filepath):
    """
    Loads pickled results as DataFrame, converts date columns
    Expects columns [refdate, team, model, target, metric_1,... metric_n]
    # TODO why not also add location_type and variable (just in case) [it'll be a bit bloated but who cares]
    filepath: str, path to file
    """
    try:
        with open(filepath, "rb") as fp:
            df = pickle.load(fp)
    except:
        df = pd.read_pickle(filepath, compression="gzip")

    if isinstance(df, list):
        df = pd.DataFrame(df[1:] if len(df) > 1 else [], columns=df[0])  # expects header in first row

    if len(df):
        df.target = series2date(df.target)
        df.refdate = series2date(df.refdate)

    return df


def pickle_results(filepath, obj):
    """
    Saves list to binary file; If obj is DataFrame, converts it to list with header in first row
    filepath: str; where to write file to
    obj: list-like or DataFrame
    """
    #if isinstance(obj, pd.DataFrame):
    #    header = list(obj.columns)
    #    obj = [header] + obj.values.tolist()
    if isinstance(obj, list):
        obj = pd.DataFrame(obj[1:], columns=obj[0])


    #with open(filepath, "wb") as fp:
    #    pickle.dump(obj, fp)

    obj.to_pickle(filepath, compression="gzip")


def get_date_range(start_date_str, days=28):
    """
    Get datetime.date objects of start and end day in 28 day interval.
    start_date_str: string of start date, i.e. '2022-10-10'
    return 2tuple of datetime.date
    """
    start_date = np.datetime64(start_date_str)

    end_date = start_date + np.timedelta64(days - 1, "D")  # first date already counts

    return start_date, end_date


def reset_results_file(fpath="results.pickle"):
    empty = [["refdate", "team", "model", "location_type", "pred_variable", "target", "location",
              "rmse", "wis", "dispersion", "underprediction", "overprediction", "mae", "mda",
              "within_50", "within_80", "within_95"]]
    pickle_results(fpath, empty)

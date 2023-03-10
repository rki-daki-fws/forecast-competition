import pickle
import datetime
import pandas as pd
import numpy as np
import requests
import os
import zipfile
from datetime import datetime
import shutil


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


def get_opendata(refdate: str) -> pd.DataFrame:
    """
    Retrieve COVID 7-day incidence on German district level for a given reference date.
    Data source is RKI OpenData repository:
    https://github.com/robert-koch-institut/COVID-19_7-Tage-Inzidenz_in_Deutschland
    """
    # assert refdate is yyyy-mm-dd
    assert len(refdate) == 10 and refdate[4] == '-' and refdate[7] == '-', "refdate should be in the format yyyy-mm-dd"
    # assert refdate is valid date. i.e. not 2022-09-35
    try:
        datetime.strptime(refdate, '%Y-%m-%d')
    except ValueError:
        raise ValueError("refdate is not a valid date")

    refdate_min = '2022-09-30'  # oldest available release
    assert datetime.strptime(refdate, '%Y-%m-%d') >= datetime.strptime(refdate_min, '%Y-%m-%d'),\
        f"refdate should be greater than or equal to {refdate_min}"

    # download data
    url = f"https://github.com/robert-koch-institut/COVID-19_7-Tage-Inzidenz_in_Deutschland/archive/refs/tags/{refdate}.zip"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to download the data")

    # save the zip file
    filename = f"{refdate}.zip"
    with open(filename, "wb") as f:
        f.write(response.content)

    # unzip the file
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall()

    # delete the zip file
    os.remove(filename)

    dir_path = f"COVID-19_7-Tage-Inzidenz_in_Deutschland-{refdate}"
    fname = "COVID-19-Faelle_7-Tage-Inzidenz_Landkreise.csv"

    df = pd.read_csv(os.path.join(dir_path, fname))  # load into memory

    shutil.rmtree(dir_path)  # clean up

    # reformat data
    # open data columns:
    #       ['Meldedatum', 'Landkreis_id', 'Bevoelkerung', 'Faelle_gesamt',
    #        'Faelle_neu', 'Faelle_7-Tage', 'Inzidenz_7-Tage']

    # repo data columns: ['location', 'target', 'value']
    df_conform = pd.DataFrame(np.zeros([df.shape[0], 3]), columns=['location', 'target', 'value'])
    df_conform['location'] = df['Landkreis_id']
    df_conform['target'] = df['Meldedatum'].astype(np.datetime64)  # TODO perhabs keep as str
    df_conform['value'] = df['Inzidenz_7-Tage'].astype(np.float16)
    #  recalculate incidence?? do we really need more than 1 decimal? nope!

    return df_conform


def calculate_7day_incidence(df: pd.DataFrame) -> pd.DataFrame:
    # Sort dataframe by date
    df = df.sort_values(by='date')

    # Compute rolling sum of cases over the last 7 days for each district
    df['cases_last_7_days'] = df.groupby('district')['cases'].rolling(window=7).sum().reset_index(0, drop=True)

    # Compute 7-day incidence
    df['7_day_incidence'] = (df['cases_last_7_days'] / df['population_size']) * 100000

    return df


def date_is_sunday(date_str: str) -> bool:
    # date_str = '2022-02-27'

    # Convert the date string to a datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')

    # Check if the day of the week is Sunday (0 = Monday, 1 = Tuesday, ..., 6 = Sunday)
    return date_obj.weekday() == 6


def greater_date(date1: str, date2: str) -> bool:
    return datetime.strptime(date1,  '%Y-%m-%d') > datetime.strptime(date2,  '%Y-%m-%d')


def smaller_date(date1: str, date2: str) -> bool:
    return datetime.strptime(date1,  '%Y-%m-%d') < datetime.strptime(date2,  '%Y-%m-%d')


def filter_last_weeks(df: pd.DataFrame, n: int=6) -> pd.DataFrame:
    today = np.datetime64('today')

    # Define a dataframe with a "date" column
    # df = pd.DataFrame({'date': ['2022-01-01', '2022-02-01', '2022-03-01', '2022-04-01'],
    #                   'value': [1, 2, 3, 4]})

    # Convert the "date" column to numpy.datetime64 type
    df['date'] = df['date'].astype(np.datetime64)

    # Filter the dataframe to include only rows with dates within the last 6 weeks
    six_weeks_ago = today - np.timedelta64(n, 'W')

    return df[df['date'] < six_weeks_ago]


if __name__ == "__main__":

    df = get_opendata("2022-09-30")
    print(df)



import pickle
import datetime
import pandas as pd
import numpy as np
import requests
import os
import zipfile
from datetime import datetime, timedelta
import shutil
import re
import glob
import warnings
from typing import Tuple
from math import ceil


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


def load_results_file(filepath):
    """
    Loads pickled results as DataFrame, converts date columns
    Expects columns [refdate, team, model, target, metric_1,... metric_n]
    filepath: str, path to file
    """
    if filepath[-4:] == ".csv":
        df = pd.read_csv(filepath)
    else:
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


def load_results(results_dir="../results", submissions_dir="../submissions",
                 glob_str="*.csv") -> pd.DataFrame:

    # glob search results dir
    files = sorted(glob.glob(f"{results_dir}/{glob_str}"))

    dfs = []
    for f in files:
        try:
            # res_team_model_locationtype_predvariable_enum.csv
            _, team, model, location_type, pred_variable, _ = f.split("_")  # unpacking might not be possible
        except:
            continue

        # check for existence of submission of that team model
        if not os.path.isdir(os.path.join(submissions_dir, team)) :
            print(f"there are no submissions for team {team}")
            continue
        if not len(glob.glob(os.path.join(submissions_dir, team, "*")+f"{'_'.join([model, location_type, pred_variable])}.parquet")):
            print(f"there are no submissions for team {team} with model {model}")
            continue

        subset_df = pd.read_csv(f)
        subset_df.target = series2date(subset_df.target)
        subset_df.refdate = series2date(subset_df.refdate)
        # add columns from file name (team, model, location_type, pred_variable
        # concat/hstack would be faster, but not sure if I used integer index of columns somewhere, so order might matter
        subset_df.insert(1, "team", subset_df.shape[0] * [team])
        subset_df.insert(2, "model", subset_df.shape[0] * [model])
        subset_df.insert(3, "location_type", subset_df.shape[0] * [location_type])
        subset_df.insert(4, "pred_variable", subset_df.shape[0] * [pred_variable])

        dfs.append(subset_df)

    if len(dfs):
        output = pd.concat(dfs, ignore_index=True)
    else:
        output = pd.DataFrame([],
                              columns=["refdate", "team", "model", "location_type", "pred_variable"] +
                                      pd.read_csv(results_dir + "/res_RKIsurv2_arima_LK_cases_1.csv").columns.to_list()[1:]
                              )

    if isinstance(output.columns, pd.core.indexes.multi.MultiIndex):
        output.columns = output.columns.get_level_values(0)
    # convert str columns to category for faster searching
    for catcol in ["team", "model", "location_type", "pred_variable"]:
        output[catcol] = output[catcol].astype('category')

    return output


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


def get_opendata(refdate: str, location_type="LK", pred_variable="cases") -> pd.DataFrame:
    """
    Retrieve COVID 7-day incidence on German district level for a given reference date.
    Data source is RKI OpenData repository:
    https://media.githubusercontent.com/media/robert-koch-institut/SARS-CoV-2-Infektionen_in_Deutschland
    Source of population data:
    https://github.com/robert-koch-institut/SARS-CoV-2-Infektionen_in_Deutschland
    """
    assert location_type in ["LK", "BL"]
    assert pred_variable in ["cases", "rvalue"]

    raw_case_data = get_data_from_day(refdate, clean_up=True)
    preprocessed = preprocess_raw_data(raw_case_data)

    # calculate 7-day incidences on LK or BL level (needs population data)
    incidences = calc_incidences(preprocessed, location_type)

    if pred_variable == "rvalue":
        rvalues = calc_rvalues(incidences, region_type=location_type)
        df_conform = df_to_competition_format(rvalues, value_column="rvalue")
    else:
        df_conform = df_to_competition_format(incidences)

    return df_conform


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    if not os.path.isfile(local_filename):
        return False
    if os.path.getsize(local_filename) > 0:
        return True
    else:
        os.remove(local_filename)
        return False


def create_url_filename(date):
    file_url = f"https://media.githubusercontent.com/media/robert-koch-institut/SARS-CoV-2-Infektionen_in_Deutschland/{date}/Aktuell_Deutschland_SarsCov2_Infektionen.csv?download=true"
    filename = f"raw_data_{date}.csv"
    return file_url, filename


class DownloadError(Exception):
    pass


def download_data_with_retries(initial_date, min_date, retries=10, retry_direction="backward"):
    assert retry_direction in ["backward", "forward"]
    # there might not be a release for the requested date, try previous or later days, up to retries if possible
    if retry_direction == "backward":
        fromdate = np.max([np.datetime64(initial_date) - np.timedelta64(retries, "D"), np.datetime64(min_date)])
        try_dates = np.arange(fromdate, initial_date,
                              dtype='datetime64[D]').tolist()[::-1]
    else:
        todate = np.min([np.datetime64(initial_date) + np.timedelta64(retries, "D"), np.datetime64('today')])
        try_dates = np.arange(initial_date, todate,
                              dtype='datetime64[D]').tolist()

    for date in try_dates:
        file_url, filename = create_url_filename(str(date))
        try:

            success = download_file(file_url, filename)
            if success:
                return filename
        except requests.exceptions.HTTPError:
            print(f"not able to download data for {str(date)}")
            continue
    else:
        raise DownloadError(
            f"After {retries} retries, there was no success downloading data from the OpenDataRepository")


def get_data_from_day(refdate, clean_up=True):
    # validate refdate
    assert len(refdate) == 10 and refdate[4] == '-' and refdate[7] == '-', \
        "refdate should be in the format yyyy-mm-dd"
    # assert refdate is valid date. i.e. not 2022-09-35
    try:
        datetime.strptime(refdate, '%Y-%m-%d')
    except ValueError:
        raise ValueError("refdate is not a valid date")

    refdate_min = '2022-06-07'  # oldest available release
    retry_direction = "backward"

    if datetime.strptime(refdate, '%Y-%m-%d') < datetime.strptime(refdate_min, '%Y-%m-%d'):
        warnings.warn(
            f"You requested a file from {refdate}. The oldest available release of the data repository is from {refdate_min}." +
            "This contains all data from the beginning of the pandemic. Continuing with that file...")
        refdate, refdate_min = refdate_min, str(np.datetime64(refdate_min) + np.timedelta64(10, "D"))
        retry_direction = "forward"

    if datetime.strptime(refdate, '%Y-%m-%d') > datetime.today():
        warnings.warn(
            f"You requested a file from the future ({refdate}). Will continue with the most recent file...")
        refdate = datetime.today().strftime("%Y-%m-%d")

    filename = download_data_with_retries(refdate, refdate_min,
                                          retry_direction=retry_direction)  # add

    df = pd.read_csv(filename)
    # process data?

    if clean_up:
        os.remove(filename)
    return df


def preprocess_raw_data(rawdata):
    # very weird data
    aggregated = rawdata.groupby(["IdLandkreis", "Meldedatum"]).sum().reset_index()
    # should automatically account for 5 corrections
    aggregated = aggregated.loc[:, ["IdLandkreis", "Meldedatum", "AnzahlFall"]]
    aggregated = zero_expand_df_all_strata(aggregated)
    return aggregated


def zero_expand_df_all_strata(df):
    """
    Expand a DataFrame to include all combinations of values of date & location,
    with zero assigned to combinations not present in the original data.

    Parameters:
    - df: DataFrame: The original DataFrame.

    Returns:
    - expanded_df: DataFrame: Expanded DataFrame with all combinations of values of date & location.
    """

    # Get all unique values for each stratum
    df.Meldedatum = df.Meldedatum.astype(np.datetime64)
    all_dates = np.arange(df.Meldedatum.min(), df.Meldedatum.max(), dtype='datetime64[D]')

    unique_locations = df.IdLandkreis.unique()

    # Create a DataFrame with all combinations of values for specified strata
    expanded_df = pd.DataFrame(index=pd.MultiIndex.from_product([all_dates, unique_locations],
                                                                names=["Meldedatum", "IdLandkreis"])
                               ).reset_index()
    expanded_df = expanded_df.sort_values(by=[ "IdLandkreis", "Meldedatum"]).reset_index(drop=True)

    # Merge the original DataFrame with the expanded DataFrame to fill in missing values
    merged_df = pd.merge(expanded_df, df, on=["Meldedatum", "IdLandkreis"], how='left')

    # Replace NaN values in 'value' column with zero
    merged_df['AnzahlFall'] = merged_df['AnzahlFall'].fillna(0)

    return merged_df


def _get_population_data_lk():
    url = 'https://raw.githubusercontent.com/robert-koch-institut/COVID-19_7-Tage-Inzidenz_in_Deutschland/2023-04-07/COVID-19-Faelle_7-Tage-Inzidenz_Landkreise.csv'
    columns = ["Landkreis_id", "Bevoelkerung"]
    df = pd.read_csv(url)
    return df.groupby(columns).size().reset_index().loc[:, columns]


def _get_population_data_bl():
    url = 'https://raw.githubusercontent.com/robert-koch-institut/COVID-19_7-Tage-Inzidenz_in_Deutschland/2023-04-07/COVID-19-Faelle_7-Tage-Inzidenz_Bundeslaender.csv'
    columns = ["Bundesland_id", "Bevoelkerung"]
    df = pd.read_csv(url)
    df = df.loc[df.Altersgruppe == "00+", :].reset_index(drop=True)
    return df.groupby(columns).size().reset_index().loc[:, columns]


def get_population_data(region_type="LK"):
    assert region_type in ["LK", "BL"]
    if region_type == "LK":
        return _get_population_data_lk()
    else:
        return _get_population_data_bl()


def calculate_incidence_from_cases(df, incidence_column="incidence", cases_column="cases",
                                   population_column="Bevoelkerung", grouping_column=None):
    df = df.copy()
    # probably needs to be done on location group
    if grouping_column is None:
        df[cases_column] = df.cases_column.shift(periods=1, fill_value=0)
        df[incidence_column] = (df[cases_column].rolling(window=7).sum() / df[population_column] * 100000).round(2)
    else:
        grouped = df.groupby([grouping_column])
        df[incidence_column] = 0.0
        case_col_iloc = df.columns.get_loc(cases_column)
        incidence_col_iloc = df.columns.get_loc(incidence_column)
        for group_label, group_indices in grouped.groups.items():
            # shift case column by one row (needs to be group-based)
            df.iloc[group_indices, case_col_iloc] = \
                df.iloc[group_indices, case_col_iloc].shift(periods=1, fill_value=0)
            df.iloc[group_indices, incidence_col_iloc] = \
                (df.iloc[group_indices, case_col_iloc].rolling(window=7).sum() /
                 df.iloc[group_indices, df.columns.get_loc(population_column)] * 100000).round(2)

    return df


def calc_incidences(df, region_type):
    # calculate 7-day incidence levels using population data
    assert region_type in ["LK", "BL"]

    pop_data = get_population_data(region_type)
    # merge population data
    if region_type == "LK":
        merged = df.merge(pop_data, left_on="IdLandkreis", right_on="Landkreis_id")
        region_column = "Landkreis_id"
    else:
        # need to aggregate first!
        region_column = "Bundesland_id"
        df["Bundesland_id"] = (df["IdLandkreis"] / 1000).astype(np.uint8)
        df = df.groupby(["Meldedatum", "Bundesland_id"]).sum().reset_index()
        merged = df.merge(pop_data, on=region_column)

    incidences = calculate_incidence_from_cases(merged, "incidence", cases_column="AnzahlFall",
                                                population_column="Bevoelkerung",
                                                grouping_column=region_column)

    return incidences.fillna(0)


def calculate_rvalues_from_cases(df, rvalue_column="rvalue", case_column="cases", grouping_column=None):
    df = df.copy()

    mask = df[case_column] == 0
    df.loc[mask, case_column] += 1

    # calculate rvalues
    if grouping_column is None:
        df[rvalue_column] = df[case_column] / df[case_column].shift(4)
    else:
        df[rvalue_column] = df[case_column] / df.groupby(grouping_column).shift(4)[case_column]

    return df.fillna(0)


def calc_rvalues(df, case_column="AnzahlFall", region_type="LK"):
    assert region_type in ["LK", "BL"]
    if region_type == "LK":
        return calculate_rvalues_from_cases(df, case_column=case_column, grouping_column="Landkreis_id")
    else:
        return calculate_rvalues_from_cases(df, case_column=case_column, grouping_column="Bundesland_id")


def df_to_competition_format(df, value_column="incidence"):
    assert value_column in ["incidence", "rvalue"]
    # clean up dataframe: remove unnecessary columns, rename columns
    if "Landkreis_id" in df.columns:
        df = df.rename(columns={value_column: "value", "Landkreis_id": "location", "Meldedatum": "target"})
    else:
        df = df.rename(columns={value_column: "value", "Bundesland_id": "location", "Meldedatum": "target"})
    return df.loc[:, ["location", "target", "value"]]


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


def filter_last_weeks(df: pd.DataFrame, n: int = 6, date_col: str = "refdate") -> pd.DataFrame:
    today = np.datetime64('today')

    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = df[date_col].astype(np.datetime64)

    # Filter the dataframe to include only rows with dates within the last n weeks
    n_weeks_ago = today - np.timedelta64(n, 'W')

    return df[df[date_col] < n_weeks_ago]


def filesize_mb(filepath):
    fstats = os.stat(filepath)  # os.path.getsize(path)
    return fstats.st_size / (1024 * 1024)


def get_cutoffs(asize, msize):
    times = ceil(asize / msize)
    return [(t * msize) / asize for t in range(times)]


def df_to_split_files(df, save_path, max_size_mb=45):

    path, basename, ending = separate_path(save_path)

    # let's try saving file first, because we don't know compression rate
    first_file = f"{path}/{basename}_{1}{ending}"

    df.to_csv(first_file, index=False, float_format='%.2f')
    full_size = filesize_mb(first_file)
    if full_size <= max_size_mb:
        return

    cutoffs = get_cutoffs(full_size, max_size_mb)
    cutoffs.append(1)  # guarantees that the last row index is always accounted for
    for i in range(len(cutoffs) - 1):
        slice_start, slice_end = int(df.shape[0] * cutoffs[i]), int(df.shape[0] * cutoffs[i + 1])
        df.iloc[slice_start:slice_end, :].to_csv(f"{path}/{basename}_{i+1}{ending}", index=False, float_format='%.2f')


def separate_path(fpath: str) -> Tuple[str, str, str]:
    """ Separates file path into directory path, file name (without trailing number) and file extension """

    directory_path, file_name_with_ext = os.path.split(fpath)
    file_name, file_extension = os.path.splitext(file_name_with_ext)

    """if "_" in file_name:
        fn_split = file_name.split("_")
        file_name = "_".join(fn_split[:-1]) # removes trailing number"""

    return directory_path, file_name, file_extension


if __name__ == "__main__":

    #df = get_opendata("2022-09-30")
    #print(df)

    res = load_results("../results", "../submissions")

    print(res.head())

    print(res.shape)
    # groupby model, team, model, location_type, variable
    grouping_columns = ["team", "model", "location_type", "pred_variable"]
    grouped = res.groupby(grouping_columns)
    for model_info, model_results in grouped:
        #print(group.loc[:, [col for col in group.columns.to_list() if col not in grouping_columns]])
        tmp_df = model_results.loc[:, [col for col in model_results.columns.to_list() if col not in grouping_columns]]
        # using group.columns.difference(grouping_columns) results in alphabetical ordering; undesired
        df_to_split_files(tmp_df, f"../results/res_{'_'.join(model_info)}.csv")

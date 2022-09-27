# Data submission instructions

In the following section, we will explain the process and requirements for a team to submit forecasts to the forecasting competition.
All forecasts will be submitted to a team-folder in the [submissions](./) folder as a parquet file following a predefined naming convention and data format. A detailed introduction of the naming convention and data format can be found below. For example, the team `testTeam` will submit all results files to `submissions/testTeam/`

To submit a forecast following steps are needed:
* create a publicly available fork from this repository
* in the submissions directory, create a new directory for your team
* place all your forecasts in this directory
* commit and push your changes
* open a pull request to the `submit` branch of the competition repository


Please make sure that
* your commits show only added files, no changed or removed ones
* all submission files adhere to the naming convention and format requirements

If these requirements are met, your pull request will be automatically accepted and your submissions will be automatically evaluated and their scores added to the leaderboard.


## Example

See [this folder](https://github.com/rki-daki-fws/forecast-competition/blob/main/examples) for an illustration of a (hypothetical) submission file. 
- `testTeam/2021-03-31_testModel_LK_cases.parquet` the hypothetical submission file
- `example.R`  a basic script showing the generation of a submission file in R


![Example Results Structure](https://github.com/rki-daki-fws/forecast-competition/blob/main/examples/example.PNG))


## Forecasting Results

### Summary

- daily cases or r-values
- time horizon of up to 28 days, starting from the date in the file name
- county (Landkreis) or states (Bundeland)
- 100 samples for each point in time and geographical entity

The teams will generate case or r-value forecasts for a time horizon of up to 28 days for either the German States (Bundesland) or Counties (Landkreis).  
Each forecast results file name should following this naming convention:

    team/YYYY-MM-DD_model_locationtype_targettype.parquet
    
where

- `YYYY` is the 4 digit year, 
- `MM` is the 2 digit month,
- `DD` is the 2 digit day,
- `team` is the name of the team, 
- `model` is the name of the model,
- `locationtype` is the kind of geographical entities for which forecasts are made (`LK` or `BL`) and
- `targettype` is either `cases` or `rvalue`. 

The date YYYY-MM-DD is the `model_date`, it corresponds to file in the `challenge-data/truth` directory which contains data leading up to that date.
That date is also the first day in the forecast period (see FAQs). 

The teams are allowed to submit forecasts for Bundesländer and Landkreise. `location` can take any of the following values:

- `LK` this corresponds to County-level (Landkreis) or
- `BL` this corresponds to State-level (Bundesland)
 

Both `team` and `model` should be less than 15 characters, 
alpha-numeric, with no spaces, underscores or hyphens.


## Forecast results file format

The file must be a parquet file with the following columns (in any order):

- `target`
- `location`
- `sample_id`
- `value`

No additional columns are allowed.

A row is a realization (sample) of a forecast of either the number of cases or the r-value (`targettype`) for the specified `target` date and `location`.  

It is generally allowed to submit incomplete data, i.e., some counties might be missing.

### `target`

Values in the `target` column must be a date in the format

    YYYY-MM-DD

This date corresponds the data snapshot date, which the forecast is based on and also represents the first day of the 28 day forecasting period. For example, if the `model_date` given in the file name is `2021-04-18`, then the value of `target` should be in the interval *[2021-04-18, 2021-05-15]*. The `value` column will hold the forecast value for that day.

### `location`

The values specified in `location` must match the definition of the `locationtype` in the results file name, such that if `LK` is the `locationtype` only valid counties (Landkreise) can be used in `location` and no states (Bundesländer). Same applies for `BL` as `locationtype`. 

#### County - Landkreis
The county ID is based on the official municipality key - Amtlicher Gemeindeschlüssel (AGS) -, which can be retrieved from the portal of [the Federal Statistical Office](https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/Administrativ/Archiv/GVAuszugQ/AuszugGV2QAktuell.html). The county ID is a concatenation of the code of the federal state (Land), the regional district (RB) and an addition for the county (LK) in this order. For a more accurate representation of Berlin, the 12 city districts are broken down as separate "Landkreise". Here, we deviate from the specifications of the AGS. The following allocation is made:

| IdLandkreis | Bezirk | IdLandkreis | Bezirk |  
| ----------- | ------ | ----------- | ------ |  
|11001 | Berlin Mitte | 11007 | Berlin Tempelhof-Schöneberg |  
|11002 | Berlin Friedrichshain-Kreuzberg | 11008 | Berlin Neukölln |  
|11003 | Berlin Pankow | 11009 | Berlin Treptow-Köpenick |  
|11004 | Berlin Charlottenburg-Wilmersdorf| 11010 | Berlin Marzahn-Hellersdorf |  
|11005 | Berlin Spandau | 11011 | Berlin Lichtenberg |  
|11006 | Berlin Steglitz-Zehlendorf | 11012 | Berlin Reinickendorf|  

#### State - Bundesland

The federal state ID equals the first two digits of the AGS and is therefore included in the country ID. The federal state IDs are listed below:

| IdBundesland | Bundsland |  
| ----------- | ------ |   
| 01 |  Schleswig-Holstein |
| 02 |  Hamburg |
| 03 |  Niedersachsen |
| 04 |  Bremen |
| 05 |  Nordrhein-Westfalen |
| 06 |  Hessen |
| 07 |  Rheinland-Pfalz |
| 08 | Baden-Württemberg |
| 09 | Bayern |
| 10 | Saarland |
| 11 | Berlin |
| 12 | Brandenburg |
| 13 | Mecklenburg-Vorpommern |
| 14 | Sachsen |
| 15 | Sachsen-Anhalt |
| 16 | Thüringen |

### `sample_id`

For each unique combination of `location` and `target`, multiple forecast `value`s will be submitted. This is meant to reflect the models uncertainty in the forecast. For 100 samples, the `sample_id` will be an integer starting at one up to 100. For a forecasting horizon of 28 days, one would essentially submit 100 different trajectories/time series of length 28 days.

### `value`

This is the actual value of the forecast. The are no restrictions other than the value of `value` being positive. 

## Parquet Files

Parquet files can easily be written from dataframes in R (`arrow` library is needed) and python (`pyarrow` and `pandas` needed). 

[writing parquet files in R](https://arrow.apache.org/docs/r/reference/write_parquet.html)

[writing parquet files in python](https://pandas.pydata.org/pandas-docs/version/1.1/reference/api/pandas.DataFrame.to_parquet.html)


## Frequently asked questions
### For which periods should I forecast?
You may submit forecasts for each of these periods:

| start date | end date   |
|------------|------------|
| 2021-04-18 | 2021-05-15 |
| 2021-04-25 | 2021-05-22 |
| 2021-05-02 | 2021-05-29 |
| 2021-05-09 | 2021-06-05 |
| 2021-05-16 | 2021-06-12 |
| 2021-05-23 | 2021-06-19 |
| 2021-05-30 | 2021-06-26 |
| 2021-06-06 | 2021-07-03 |
| 2021-06-13 | 2021-07-10 |
| 2021-06-20 | 2021-07-17 |
| 2021-06-27 | 2021-07-24 |
| 2021-07-04 | 2021-07-31 |
| 2021-07-11 | 2021-08-07 |
| 2021-07-18 | 2021-08-14 |
| 2021-07-25 | 2021-08-21 |


### For how many days should I forecast?
For each forcast that you submit, please forcast a maximum of 28 days into the future, starting from the snapshot date. Depending on your
model, you might want to forecast less than 28 days. In this case, you don't need to forecast consecutive days.

### Why was my submission pull request not automatically merged?
There might be a number of reasons:
* Your submission file did not adhere to the naming convention
* Your submission file did not adhere to the required data format
* Besides new submissions files your pull request also included changes to existing files

While this might be inconvenient,  rest assured that your submission will not go unnoticed and we will look into it and 
possibly merge manually.
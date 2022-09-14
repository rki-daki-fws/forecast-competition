# Data submission instructions

In the following section we will explain the process and requirements for a team to submit forcasts to the forecasting competition.
All forecasts will be submitted to a team-folder in the [submissions](./) folder as a parquet file follwing a predefined naming convention and data format. A detailed introduction of the naming convention and data format can be found below. For example the team `testTeam` will submit all results files to `submissions/testTeam/`

To submit a forecast follwing steps are needed:
* create a publicly available fork from this repository
* in the submissions directory, create a new directory for your team
* place all your forecasts in this directory
* commit and push your changes
* open a pull request to the submit branch of the competition repository


Please make sure that
* your commits show only added files, no changed or removed ones
* all submission files adhere to the naming convention and format requirements
If these requirements are met, your pull request will be automatically accepted and your submissions will be automatically evaluated and their scores added to the leaderboard.


## Example

See [this folder](https://github.com/rki-daki-fws/forecast-competition/blob/main/examples) for an illustration of a (hypothetical) submission file. 
- `2021-03-31_testTeam_testModel_LK_cases.parquet` the hypothetical submission file
- `example.R`  a basic script showing the generation of a submission file in R


![Example Results Structure](https://github.com/rki-daki-fws/forecast-competition/blob/main/examples/example.PNG))


## Forecasting Results

### Summary

- daily cases or r-values
- time horizon of 28 days
- county (Landkreis) or states (Bundeland)
- 100 samples for each point in time and geographical entity

The teams will generate case or r-value forecasts for a time horizon of 28 days for either the German States (Bundesland) or Counties (Landkreis).  
Each forecast results file name should following this naming convention:

    YYYY-MM-DD_team_model_locationtype_targettype.parquet
    
where

- `YYYY` is the 4 digit year, 
- `MM` is the 2 digit month,
- `DD` is the 2 digit day,
- `team` is the name of the team, 
- `model` is the name of the model,
- `locationtype` is the kind of geographical entities for which forecasts are made (`LK` or `BL`) and
- `targettype` is either `cases` or `rvalue`. 

The date YYYY-MM-DD is the `model_date`, the latest date for which data is being used to fit the forecasting model. 

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

It is generally allowed to submit incomplete data i.e. some counties might be missing.

### `target`

Values in the `target` column must be a date in the format

    YYYY-MM-DD

This date corresponds to the forecasting date. For example if the `model_date` given in the file name is `2021-04-01` and the date in `target` is `2021-04-02` then `value` will hold the 1-day-ahead forecast.

### `location`

The values specified in `location` must match the definition of the `locationtype` in the results file name, such that if `LK` is the `locationtype` only valid counties (Landkreise) can be used in `location` and no states (Bundesländer). Same applies for `BL` as `locationtype`. 

#### County - Landkreis
The county ID is based on the official municipality key/Amtlicher Gemeindeschlüssel (AGS), which can be retrieved from the portal of [the Federal Statistical Office](https://www.destatis.de/DE/Themen/Laender-Regionen/Regionales/Gemeindeverzeichnis/Administrativ/Archiv/GVAuszugQ/AuszugGV2QAktuell.html). The county ID results from the code of the federal state (Land), the regional district (RB) and the county (LK). For a more accurate representation of Berlin, the 12 city districts are broken down as separate "Landkreise". Here we deviate from the specifications of the AGS. The following allocation is made:

| IdLandkreis | Bezirk | IdLandkreis | Bezirk |  
| ----------- | ------ | ----------- | ------ |  
|11001 | Berlin Mitte | 11007 | Berlin Tempelhof-Schöneberg |  
|11002 | Berlin Friedrichshain-Kreuzberg | 11008 | Berlin Neukölln |  
|11003 | Berlin Pankow | 11009 | Berlin Treptow-Köpenick |  
|11004 | Berlin Charlottenburg-Wilmersdorf| 11010 | Berlin Marzahn-Hellersdorf |  
|11005 | Berlin Spandau | 11011 | Berlin Lichtenberg |  
|11006 | Berlin Steglitz-Zehlendorf | 11012 | Berlin Reinickendorf|  

#### State - Bundesland

Like the county Id is the State Id based on the AGS and is just the code of the federal state(Land).

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

For each unique combination of `location` and `target` multiple forecast `value`s will be submitted. This is meant to reflect the models uncertainty in the forecast. For 100 samples the `sample_id` will be an integer starting at one upto 100. For a forecasting horizon of 28 days one would essentially submit 100 different trajectories/time series of length 28 days.

### `value`

This is the actual value of the forecast. The are no restrictions other than the value of `value` being positive. 

## Parquet Files

Parquet files can easily be written from dataframes in R (`arrow` library is needed) and python (`pyarrow` and `pandas` needed). 

[writing parquet files in R](https://arrow.apache.org/docs/r/reference/write_parquet.html)

[writing parquet files in python](https://pandas.pydata.org/pandas-docs/version/1.1/reference/api/pandas.DataFrame.to_parquet.html)

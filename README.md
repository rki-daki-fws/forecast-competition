# DAKI-FWS forecast competition

This competition is about forecasting the development of the COVID19 pandemic in Germany, as part of the DAKI FWS project.

We invite all participants to upload their forecast data until **October 9th 2022**.
The submissions will first be automatically evaluated by a simple metric of fit (RSME) and the results will be displayed in a public leaderboard.

After all participants submitted their forecasts, we will analyse the submissions more in-depth with more advanced metrics.
We thank you all for participating!

## Data for training and evaluation
You can find data for training and evaluation purposes in the `challenge-data` directories `truth` and `evaluation`.

The target variables are case numbers of infections and the basic reproductive number (or R-value). 

You will notice that the files in the `challenge-data/truth` directory have a date associated to them in their filename. 
That is the time the data snapshot was taken and the files contain only data until the previous day.
As time progresses, past values can change due to reporting delay. That is why we provide you with the different data snapshots.  
You can use these files to train your models. Just make sure that you don't use data from the future in training to predict values in the past.
I.e. to predict infection cases for the period of April 18th 2021 to May 15th 2021 you would only use the file `2021-04-18_LK_cases.csv`.
If you wanted to forecast the 28 day period starting on July 4th 2021, you could train your model with only the snapshot data of that day.
Alternatively, you could use all prior snapshot files, to account for reporting delays in your model.

The data in the `challenge-data/evaluation` directory can be perceived as the final state of the data and will be used for grading the forecasting performance of the submissions.
Please use these only for a final evaluation of your models, not for training.

## How to take part

To take part in the forecasting competition, please submit your predictions until October 9th 2022 via a pull request to this repo's `submit` branch.
You can upload forecasts for any of the files in the `challenge-data/truth` directory. Each forecast should predict values for up to 28 days, starting at the date in the file name.

For more detailed instructions, please see [here](https://github.com/rki-daki-fws/forecast-competition/blob/main/submissions/README.md).

## Leaderboard

![Top ten leaderboard positions](https://github.com/mlbach/example-competition/blob/main/leaderboard_snapshot.png)
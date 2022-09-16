# DAKI-FWS forecast competition

This competition is about forecasting the development of the COVID19 pandemic in Germany, as part of the DAKI FWS project.

In the first phase of the competition, we invite all participants to upload their forecast data which will be automatically evaluated by a simple
metric of fit (RSME). A public leaderboard will display the results all submissions scored.

In the next phase, we will analyse the submissions more in-depth with more advanced metrics.
We thank you all for participating!

## Data for training and evaluation
You can find data for training and evaluation purposes in the `challenge-data` directories `truth` and `evaluation` respectively.

The target variables are currently case numbers of infections and the basic reproductive number (or R-value). In the future
this might be expanded to other target variables, such as epidemiological outbreak signals.

You will notice that the files in the `challenge-data/truth` directory have a date associated to them in their filename. 
That is the time the data snapshot was taken. In their contents however, the date ranges between the files overlap, 
though with possible different target values.
As time progresses, past values can change due to reporting delay. That is why we provide you with different files for 
training your models.  

The data in the `challenge-data/evaluation` directory will be used for grading the forecasting performance of the submissions.
Please use these only for a final evaluation of your models, not for training.

## How to take part

To take part in the forecasting competition, please follow these [instructions](https://github.com/rki-daki-fws/forecast-competition/blob/main/submissions/README.md).

## Leaderboard

![Top ten leaderboard positions](https://github.com/mlbach/example-competition/blob/main/leaderboard_snapshot.png)
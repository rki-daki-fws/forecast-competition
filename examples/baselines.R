# author: Philip Oedi
# Date: 2022-09-05
# Script for the creation of a simple forecasting model
# and transformation to required format for the submission of results

library(dplyr)
library(readr)
library(tidyr)
library(fable)
library(tsibble)
library(lubridate)
library(arrow)
library(parallel)
library(stringr)

# make sure working directory is correct, so relative paths work
wd <- getwd()
if(basename(wd) != "examples"){
  setwd(file.path(wd, "examples"))
}

get_opendata <- function(referance_date){
  url <- paste("https://github.com/robert-koch-institut/COVID-19_7-Tage-Inzidenz_in_Deutschland/archive/refs/tags/",
                referance_date, ".zip", sep="")

  temp <- "./tempfile.zip"
  download.file(url,temp, mode="wb")
  to_extract <- paste("COVID-19_7-Tage-Inzidenz_in_Deutschland-", referance_date,
                      "/COVID-19-Faelle_7-Tage-Inzidenz_Landkreise.csv", sep="")
  data <- read_csv(unz(temp, to_extract))
  unlink(temp)
  data[-c(3:6)] %>% rename(
    target = Meldedatum,
    location = Landkreis_id,
    value = "Inzidenz_7-Tage") %>%
    tidyr::complete(location, target, fill = list(value = 0)) %>%
    mutate(location = as.numeric(location)) %>%
    as_tsibble(key = location, index = target)  # target is date
}



make_forecasts <- function(team, forecast_start) {

  location_type <- "LK"

  # retrieve openData
  data <- get_opendata(forecast_start)

  # fitting baseline models
  train <- data %>%
    select(target, location, value) %>%
    filter(target < forecast_start)

  model <- train %>%
    model(
      etsANN = ETS(value ~ error("A") + trend("N") + season("N")),
      etsAAN = ETS(value ~ error("A") + trend("A") + season("N")),
      arima = ARIMA(),
      etsAAA = ETS(value ~ error("A") + trend("A") + season("A")))

  # generate 100 sample trajectories for 28 days
  data_forecast <- model %>%
    generate(h = "28 days", times = 100)

  # rename columns to fit results format
  data_submit <- data_forecast %>%
    as_tibble() %>%
    rename(
      target = target,
      location = location,
      sample_id = .rep,
      value = .sim,
      model = .model) %>%
    #  select(-c(.model)) %>%
    mutate(value = ifelse(value < 0,0,value))

  # write files
  for (m in c("etsANN","etsAAN","arima","etsAAA")){
    output_name <- paste(forecast_start,m,location_type,"cases", sep="_")
    output_name <- paste0(output_name, ".parquet")
    #dir.create(team)
    fp <- file.path("..", "submissions", team, output_name)
    data_submit %>%
      filter(model == m) %>%
      select(-c(model)) %>%
      write_parquet(fp)

    if(!file.exists(fp))
    {
      stop("File could not be created!") # aborting pipeline!
    }
  }
}

# metadata
team_name <- "RKIsurv2"
forecast_start <- Sys.Date() # as.Date("2021-04-18")
location_type <- "LK"

make_forecasts(team_name, forecast_start)

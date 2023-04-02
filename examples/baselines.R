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
library(httr)
library(comprehenr)

# make sure working directory is correct, so relative paths work
wd <- getwd()
if(basename(wd) != "examples"){
  setwd(file.path(wd, "examples"))
}


get_opendata <- function(referance_date){
  success = FALSE
  temp <- "./tempfile.zip"
  
  while(success == FALSE){
    url <- paste("https://github.com/robert-koch-institut/COVID-19_7-Tage-Inzidenz_in_Deutschland/archive/refs/tags/",
                  referance_date, ".zip", sep="")
    
    res <- GET(
      url = url,
      write_disk(temp, overwrite=TRUE))#,
      #verbose()) # deals with redirects
    
    if(res$status_code == 200){
      success = TRUE
    }
    else{
      #print(h$value())
      print(res)
      print(paste("Archive file not available for", referance_date, ".\n Trying previous day."))
      referance_date = as.Date(referance_date) - days(1)
    }
    
  }
  
  #download.file(url,temp, mode="wb", extra="-L")  # doesn't deal with redirects
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

  if(forecast_start >= "2022-10-01"){
    # retrieve openData
    data <- get_opendata(forecast_start)    
  }
  else{
    data <- read_csv("../challenge-data/evaluation/2022-10-02_LK_cases.csv") %>%
      as_tsibble(key = location, index = target)
  }

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


get_sundays <- function(year){
  start <- seq(as.Date(paste0(year, "-01-01")), 
               as.Date(paste0(year, "-01-07")),
               by = 1)
  sundays <- seq(start[which(as.numeric(format(start,"%w")) == 0)],
                 as.Date(paste0(year, "-12-31")), by = 7)
  # only up to now
  sundays[sundays <= Sys.Date()]
}


already_predicted <- function(){
  
  files <- list.files("../submissions/RKIsurv2")
  dates <- to_vec(for(f in files) str_split(f,"_")[[1]][1])
  #unique(dates)
}


#ttwo <- get_sundays("2022")
#tthree <- get_sundays("2023")
#predicted <- already_predicted()
#remaining <- setdiff(as.character(append(ttwo, tthree)), predicted) # x - y

# metadata
team_name <- "RKIsurv2"
forecast_start <- Sys.Date() # as.Date("2021-04-18")
location_type <- "LK"

make_forecasts(team_name, forecast_start)

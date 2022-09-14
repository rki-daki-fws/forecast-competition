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

data <- read_csv(file.path("challenge-data","incidences_reff_2021-03-31.csv")) %>%
  as_tsibble(key = county, index = date)

# metadata
team_name <- "testTeam"
model_name <- "testModel"
forecast_start <- as.Date("2021-04-01")
model_date <- forecast_start - 1
location_type <- "LK"

# fitting an ets
train <- data %>%
  select(date, county, cases) %>%
  filter(date < forecast_start)

model <- train %>% 
  model(ets = ETS(cases)) 

# generate 100 sample trajectories for 28 days
data_forecast <- model %>%
  generate(h = "28 days", times = 100) 

# rename columns to fit results format
data_submit <- data_forecast %>%
  as_tibble() %>%
  rename(
    target = date,
    location = county,
    sample_id = .rep,
    value = .sim) %>%
  select(-c(.model)) %>%
  mutate(value = ifelse(value < 0,0,value))

# YYYY-MM-DD_team_model_locationtype_targettype.parquet
# 2021-03-31_testTeam_testModel_LK_cases.parquet
output_name <- paste(model_date,team_name,model_name,location_type,"cases", sep="_")
output_name <- paste0(output_name, ".parquet")
write_parquet(data_submit, file.path("examples",output_name))

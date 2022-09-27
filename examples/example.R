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

data <- read_csv(file.path("..","challenge-data","truth", "2021-04-18_LK_cases.csv")) %>%
  as_tsibble(key = location, index = target)  # target is date

# metadata
team_name <- "testTeam"
model_name <- "testModel"
forecast_start <- as.Date("2021-04-18")
model_date <- forecast_start
location_type <- "LK"

# fitting an ets
train <- data %>%
  select(target, location, value) %>%
  filter(target < forecast_start)

model <- train %>% 
  model(ets = ETS(value)) 

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
    value = .sim) %>%
  select(-c(.model)) %>%
  mutate(value = ifelse(value < 0,0,value))

# team/YYYY-MM-DD_model_locationtype_targettype.parquet
# testTeam/2021-04-18_testModel_LK_cases.parquet
dir.create(team_name)
output_name <- paste(model_date,model_name,location_type,"cases", sep="_")
output_name <- paste0(output_name, ".parquet")
write_parquet(data_submit, file.path(team_name, output_name))

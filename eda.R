library(ggplot2)
library(readr)
library(dplyr)
library(tidyr)
library(DT)



getwd()

app_events <- read_csv('app_events.csv')
app_events %>% glimpse

##seeing if there are na values in the columns of app_events
sapply(app_events, function(x)prop.table(table(is.na(x))))


# Data Inspection

# Predicting the season a quote from friends is from

# loading packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(friends)

# handling common conflicts
tidymodels_prefer()

# quick glance over the dataset ----
friends

friends |> skimr::skim_without_charts()
# every column name is types correctly and is of the right type
# some missingness issues w/ speaker and waaaay too many levels

# brief cleaning ----
friends <- friends |> mutate(
  text = as.factor(text),
  # changed to factor because limited options it can be
  season = as.factor(season)
)

# splitting and folding data ----
# splitting data
set.seed(33726)
friends_split <- friends |> initial_split(prop = 0.8, strata = season)
friends_train <- training(friends_split)
friends_test <- testing(friends_split)

# folding data
set.seed(1875)
friends_folds <- friends_train |> vfold_cv(v = 10, repeats = 10, strata = season)

# writing out results ----
save(friends_split, file = here("friends/data_splits/friends_split.rda"))
save(friends_train, file = here("friends/data_splits/friends_train.rda"))
save(friends_test, file = here("friends/data_splits/friends_test.rda"))
save(friends_folds, file = here("friends/data_splits/friends_folds.rda"))

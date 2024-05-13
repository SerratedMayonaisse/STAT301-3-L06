# Friends NLP Prediction Problem ----
# Fit null model

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(doParallel)

# Parallel Processing ----
num_cores <- parallel::detectCores(logical = TRUE)
# create a cluster object and then register:
cl <- makePSOCKcluster(num_cores - 1)
registerDoParallel(cl)

# Handle common conflicts
tidymodels_prefer()

# load required objects ----
load(here("friends/recipes/friends_recipe.rda"))
load(here("friends/data_splits/friends_folds.rda"))
load(here("friends/data_splits/friends_train.rda"))

# model specification ----
null_spec <-
  null_model() |>
  set_mode("classification") |>
  set_engine("parsnip")

# workflow ----
null_wflow <-
  workflow() |>
  add_model(null_spec) |>
  add_recipe(friends_recipe)

# Tuning/fitting ----
null_res <-
  null_wflow |>
  fit_resamples(
    resamples = friends_folds,
    control = control_resamples(save_pred = TRUE, save_workflow = TRUE)
  )

# set seed
set.seed(6477)
final_null_train <- fit(null_wflow, data = friends_train)

# Write out results & workflow ----
save(null_res, file = here("friends/results/null_res.rda"))
save(final_null_train, file = here("friends/results/final_null_train.rda"))

# stopping parallel processing
stopCluster(cl)

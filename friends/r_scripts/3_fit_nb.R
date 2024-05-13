# Friends NLP Prediction Problem ----
# Fit null model

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(doParallel)
library(discrim)

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
nb_spec <-
  naive_Bayes() |>
  set_mode("classification") |>
  set_engine("naivebayes")

# workflow ----
nb_wflow <-
  workflow() |>
  add_model(nb_spec) |>
  add_recipe(friends_recipe)

# Tuning/fitting ----
nb_res <-
  nb_wflow |>
  fit_resamples(
    resamples = friends_folds,
    control = control_resamples(save_pred = TRUE, save_workflow = TRUE)
  )

# set seed
set.seed(6477)
final_nb_train <- fit(nb_wflow, data = friends_train)

# Write out results & workflow ----
save(nb_res, file = here("friends/results/nb_res.rda"))
save(final_nb_train, file = here("friends/results/final_nb_train.rda"))

# stopping parallel processing
stopCluster(cl)
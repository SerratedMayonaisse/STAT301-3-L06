# Friends NLP Pred ----
# Assess model

# Load package(s) and data ----
library(tidymodels)
library(tidyverse)
library(here)

# Handle common conflicts
tidymodels_prefer()

# Load testing data
load(here("friends/data_splits/friends_test.rda"))

# Load trained ensemble model info
load(here("friends/results/nb_res.rda"))
load(here("friends/results/null_res.rda"))
load(here("friends/results/final_null_train.rda"))
load(here("friends/results/final_nb_train.rda"))

# Assessing naive bayes trained model ----
nb_metrics <- collect_metrics(nb_res) |> 
  mutate(
    model = "naive bayes"
  )
nb_metrics

# Assessing naive bayes model with test data ----
nb_pred <-
  friends_test |> 
  select(season) |> 
  bind_cols(predict(final_nb_train, friends_test))
nb_pred

nb_conf_mat <- conf_mat_resampled(nb_res, tidy = FALSE) |> 
  autoplot(type = "heatmap")

nb_accuracy <- accuracy(nb_pred, truth = season, .pred_class) |> 
  mutate(model = "naive bayes")

# Assessing null trained model ----
null_metrics <- collect_metrics(null_res)
training_metrics <- null_metrics |> mutate(
  model = "null"
) |> full_join(nb_metrics) |> 
  knitr::kable() |> 
  kableExtra::kable_paper()
training_metrics

# Assessing  null model with test data ----
null_pred <-
  friends_test |> 
  select(season) |> 
  bind_cols(predict(final_null_train, friends_test))
null_pred

final_accuracy <- accuracy(null_pred, truth = season, .pred_class) |> 
  mutate(model = "null") |> 
  full_join(nb_accuracy) |> 
  arrange(.estimate) |> 
  knitr::kable() |> 
  kableExtra::kable_paper()
final_null_accuracy

conf_mat_null <- conf_mat_resampled(null_res, tidy = FALSE) |> 
  autoplot(type = "heatmap")
conf_mat_null

# saving stuff
save(final_null_accuracy, file = here("friends/figures/final_accuracy.rda"))
save(conf_mat_null, file = here("friends/figures/conf_mat_null.rda"))
save(training_metrics, file = here("friends/figures/training_metrics.rda"))
save(nb_conf_mat, file = here("friends/figures/nb_conf_mat.rda"))

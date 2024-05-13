# Friends NLP Problem ----

# Set basic feature engineering/recipe

# Load package(s)
library(tidymodels)
library(tidyverse)
library(here)
library(textrecipes)

# Handle common conflicts
tidymodels_prefer()

# load training data
load(here("friends/data_splits/friends_train.rda"))
friends_train |> colnames()

## Build general recipe (featuring eng.) ----
friends_recipe <- recipe(season ~ text, data = friends_train) |> 
  step_tokenize(text) |> 
  step_tokenfilter(text, max_tokens = 1e3) |> 
  step_tfidf(text)

# checking recipe
friends_recipe |>
  prep(friends_train) |>
  bake(new_data = NULL)

# writing out results ----
save(friends_recipe, file = here("friends/recipes/friends_recipe.rda"))


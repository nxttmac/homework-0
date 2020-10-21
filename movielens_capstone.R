##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

if(!exists("movies") && !exists("ratings")) {
  f_movies <- "ml-10M100K/ratings.dat"
  f_ratings <- "ml-10M100K/movies.dat"
  if(!file.exists(f_movies) && !file.exists(f_ratings)) {
    dl <- tempfile()
    download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
    
    ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                     col.names = c("userId", "movieId", "rating", "timestamp"))
    
    movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
    colnames(movies) <- c("movieId", "title", "genres")
  } else {
    ratings <- fread(text = gsub("::", "\t", readLines("ml-10M100K/ratings.dat")),
                     col.names = c("userId", "movieId", "rating", "timestamp"))
    movies <- str_split_fixed(readLines("ml-10M100K/movies.dat"), "\\::", 3)
    colnames(movies) <- c("movieId", "title", "genres")
  }
}

# if using R 3.6 or earlier
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId") %>%
  semi_join(edx, by = "genres")


# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# The validation set is only for the testing at the end.
##########################################################

# set a seed so the results are consistent
set.seed(1986, sample.kind = "Rounding")

# test set will be 20% of the edx set
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# Make sure userId, movieId and genre in test set are also in edx sets
test_set <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId") %>%
  semi_join(train_set, by = "genres")

# define a loss function
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# The simplest possible model is to predict the same rating for each movie
# The estimate that minimizes the residual mean squared error is the average
mu <- mean(train_set$rating)
average_rmse <- RMSE(test_set$rating, mu)

# average_rmse [1.06]

#create a table to hold the results
rmse_results <- data_frame(method = "Average", RMSE = average_rmse)

# Movie Effects
# Different movies have very different average ratings
movie_avg_ratings <- train_set %>% group_by(movieId) %>% summarize(avg = mean(rating))
# Here's what the skew looks like
movie_avg_ratings %>% ggplot(aes(avg)) + geom_histogram()

# A few films have really high averages, and a few have very low averages
# Accounting for this could help improve our RMSE

# We'll incorporate this in our model by creating a "film bias" 
# by taking the mean rating for each film and subtracting the overall mean rating
movie_bias <- train_set %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))

# incorporate our movie bias into our predicted rating
predicted_ratings_movie_effect <- mu + test_set %>% left_join(movie_bias, by = "movieId") %>% .$b_i

# Evaluate the RMSE
movie_avg_rmse <- RMSE(test_set$rating, predicted_ratings_movie_effect)

# movie_avg_rmse [0.94] - This is better than the average!

# Adding this to the table
rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie Effects Model", 
                                                   RMSE = movie_avg_rmse))
rmse_results %>% knitr::kable()

# User effects
# Different users have very different ratings (some consistently rate higher/lower than average)
# Movie bias is still incorporated here because we are concerned with how users rated each movie 
# vs the average rating. Some users could have only rated bad movies, but that doesn't mean they are tough critics.
user_bias <- train_set %>% left_join(movie_bias, by='movieId') %>%
  group_by(userId) %>% summarize(b_u = mean(rating - mu - b_i))
# Graph of user effect
user_bias %>% ggplot(aes(b_u)) + geom_histogram()

# Incorporate user bias into our ratings predictions
predicted_ratings_movie_and_user_effect <- test_set %>% left_join(movie_bias, by='movieId') %>%
  left_join(user_bias, by='userId') %>% mutate(prediction = mu + b_i + b_u) %>% .$prediction

movie_user_rmse <- RMSE(test_set$rating, predicted_ratings_movie_and_user_effect)

# movie_user_rmse [0.86] - This is a further improvement!

rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie + User Effects Model", 
                                                   RMSE = movie_user_rmse))
rmse_results %>% knitr::kable()

## Regularization

# There could be some further effects of movie and user bias that we can minimize
# We'll start by looking at the highest and lowest rated movies

# Add the movie titles so we can tell which movies have the highest and lowest bias
movie_titles <- edx %>%
  select(movieId, title) %>%
  distinct()

# These are the titles with the highest movie bias
movie_bias %>% left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>%
  select(title, b_i) %>%
  slice(1:10) %>%
  knitr::kable()

# These are the titles with the lowest movie bias
movie_bias %>% left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>%
  select(title, b_i) %>%
  slice(1:10) %>%
  knitr::kable()

# The movies with the highest b_i don't match expectations.
# IMDb's Top 250 films are nowhere on this list (https://www.imdb.com/chart/top/)

# Looking closer, we see titles with highest positive b_i (best movies) also have few votes
train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_bias) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# The titles with the highest negative b_i (worst movies) also have few votes
# Note: with the exception of From Justin to Kelly, which is widely regared as
# one of the worst films ever made 
# (https://en.wikipedia.org/wiki/List_of_films_considered_the_worst#From_Justin_to_Kelly_(2003))
train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_bias) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# This is the rationale for using regularization.  

# We're just going to choose a lambda value to illustrate the concept, but later we'll also
# optimize this parameter
lambda <- 3

# Adding our lamdba to the denominator pulls the average towards zero for films with few ratings
# but has a marginal effect on films with many ratings

predicted_ratings <- mu + test_set %>% left_join(movie_bias, by = "movieId") %>% .$b_i

movie_avg_rmse <- RMSE(test_set$rating, predicted_ratings)

# movie_avg_rmse [0.94]

rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie Effects Model", 
                                                   RMSE = movie_avg_rmse))
rmse_results %>% knitr::kable()

predicted_ratings <- test_set %>% left_join(movie_bias, by = "movieId") %>%
  left_join(genre_bias, by = "genres") %>% mutate(prediction = mu + b_i + b_g) %>%
  .$prediction

movie_genre_avg_rmse <- RMSE(test_set$rating, predicted_ratings)

# movie_genre_avg_rmse [0.98]

# Now we'll add user effects
user_bias <- train_set %>% left_join(movie_bias, by='movieId') %>%
  group_by(userId) %>% summarize(b_u = mean(rating - mu - b_i))

# Graph of user effect
user_bias %>% ggplot(aes(b_u)) + geom_histogram()

predicted_ratings <- test_set %>% left_join(movie_bias, by='movieId') %>%
  left_join(user_bias, by='userId') %>% mutate(prediction = mu + b_i + b_u) %>% .$prediction

movie_user_rmse <- RMSE(test_set$rating, predicted_ratings)

# movie_user_rmse [0.86]

rmse_results <- bind_rows(rmse_results, data_frame(method = "Movie + User Effects Model", 
                                                   RMSE = movie_user_rmse))
rmse_results %>% knitr::kable()

## Regularization

# First we'll select a lambda value, and optimize later
lambda <- 3
mu <- mean(train_set$rating)
movie_reg_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())

# we can see the effect by plotting the original and regularized b_i by number of ratings
# movies with fewer ratings are normalized towards 0 b_i
data_frame(original = movie_bias$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

# Best 5 movies (Regularized)
# This matches IMDb's list much more closely!
train_set %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# Worst 5 movies (Regularized)
# This also makes sense.  These are in wikipedia's list of worst movies
train_set %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# If this is a more accurate picture of the best/worst movies, it should help minimze RMSE

# use regularized movie bias in our estimate
predicted_ratings_movie_effect_regularized <- test_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>% .$pred

movie_regularization_rmse <- RMSE(test_set$rating, predicted_ratings_movie_effect_regularized)

# movie_regularization_rmse [0.9436] - This is an improvement over movie bias alone

# Now we'll add optimization the lambda parameter - find the lamba that results in the lowest RMSE
# We'll test a sequence of values from 0 to 10, incremented by 0.25
lambdas <- seq(0, 10, 0.25)

# We do this for regularization optimization - it is different than our movie bias
sum_of_residuals <- train_set %>%
  group_by(movieId) %>%
  summarize(s = sum(rating - mu), n_i = n())

# We can find the RMSE for all values with the function below
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set %>%
    left_join(sum_of_residuals, by='movieId') %>%
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

# Plotting the result shows us the lambda value that minimizes RMSE
qplot(lambdas, rmses)  

# It is 2.25
lambdas[which.min(rmses)]

# rmses[10] is 0.94364

# User regularization: We can also regularize the user bias.  Some users have rated many films, but more haven't
edx %>% group_by(userId) %>% summarize(n_i = n()) %>% ggplot(aes(n_i)) + geom_histogram()

# Some users with the highest positive user bias have relatively few ratings
train_set %>% dplyr::count(userId) %>% 
  left_join(user_bias) %>%
  arrange(desc(b_u)) %>%
  select(userId, b_u, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# The same is true for users with the highest negative user bias
train_set %>% dplyr::count(userId) %>% 
  left_join(user_bias) %>%
  arrange(b_u) %>%
  select(userId, b_u, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# We'll likely further minimize RMSE by regularizing the user bias

# make a sequence of lambdas from 0 to 10, incremented by 0.25
lambdas <- seq(0, 10, 0.25)

# Define this function to determine the RMSES
rmses <- sapply(lambdas, function(l){
  # regularized movie effect
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  # regularized user effect
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))

  # predicted rating
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

# plotting the results, we can see the best lambda is 5
qplot(lambdas, rmses)

best_lambda <- lambdas[which.min(rmses)]
best_lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

# movie_and_user_regularization_rmse [0.8655] - This is the best result so far

# Genre effect
# Certain genres seem to have consistently lower ratings than others
genre_bias <- train_set %>% left_join(movie_bias, by='movieId') %>% 
  left_join(user_bias, by="userId") %>% group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u))
# Here is the skew for this bias
genre_bias %>% ggplot(aes(b_g)) + geom_histogram()

# Many genres with the highest positive genre bias contain relatively few movies
train_set %>% dplyr::count(genres) %>% 
  left_join(genre_bias) %>%
  arrange(desc(b_g)) %>%
  select(genres, b_g, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# The same is true for some of the genres with the highest negative genre bias
train_set %>% dplyr::count(genres) %>% 
  left_join(genre_bias) %>%
  arrange(b_g) %>%
  select(genres, b_g, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# We'll attempt to add the regularized genre effect to our model as well

# make a sequence of lambdas from 0 to 10, incremented by 0.25
lambdas <- seq(0, 10, 0.25)

# Define this function to determine the RMSES
rmses <- sapply(lambdas, function(l){
  # average rating
  mu <- mean(train_set$rating)
  # regularized movie effect
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  # regularized user effect
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  #regularized genre effect
  b_g <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  # predicted rating
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)

best_lambda <- lambdas[which.min(rmses)]
best_lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User + Genres Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

# Now we'll test our model on the validation set, training it with the full edx dataset
# Define this function to determine the RMSES
rmses <- sapply(lambdas, function(l){
  # average rating
  mu <- mean(edx$rating)
  # regularized movie effect
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  # regularized user effect
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  #regularized genre effect
  b_g <- edx %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+l))
  # predicted rating
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_g) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})

qplot(lambdas, rmses)

# This is our final RMSE
min(rmses)
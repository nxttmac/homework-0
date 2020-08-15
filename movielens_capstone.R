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

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later (I have R 3.6)
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
#                                           title = as.character(title),
#                                           genres = as.character(genres))

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
  semi_join(edx, by = "userId")

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

# Different movies have very different ratings
movie_avg_ratings <- train_set %>% group_by(movieId) %>% summarize(avg = mean(rating))
# Here's what the skew looks like
movie_avg_ratings %>% ggplot(aes(avg)) + geom_histogram()

# A few films have really high averages, and a few have very low averages
# Accounting for this could help improve our RMSE

# We'll incorporate this in our model by creating a "film bias" 
# by taking the mean rating for each film and subtracting the overall mean rating
movie_bias <- train_set %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu))

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

# Looking closer, we see titles with highest b_i also have few votes
train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_bias) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

train_set %>% dplyr::count(movieId) %>% 
  left_join(movie_bias) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()


movie_bias

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

# TODO: add rationale for regularization
lambda <- 3
mu <- mean(train_set$rating)
movie_reg_avgs <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n())

data_frame(original = movie_bias$b_i, 
           regularlized = movie_reg_avgs$b_i, 
           n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

# Best 5 movies (Regularized)
train_set %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

# Worst 5 movies (Regularized)
train_set %>%
  dplyr::count(movieId) %>% 
  left_join(movie_reg_avgs) %>%
  left_join(movie_titles, by="movieId") %>%
  arrange(b_i) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable()

predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by='movieId') %>%
  mutate(pred = mu + b_i) %>%
  .$pred

movie_regularization_rmse <- RMSE(test_set$rating, predicted_ratings)

lambdas <- seq(0, 10, 0.25)

mu <- mean(train_set$rating)
just_the_sum <- train_set %>%
  group_by(movieId) %>%
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set %>%
    left_join(just_the_sum, by='movieId') %>%
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

# TODO: Rationale for user rating regularization - how many ratings does the average user have?
edx %>% group_by(userId) %>% summarize(n_i = n()) %>% ggplot(aes(n_i)) + geom_histogram()

lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(rmses)))
rmse_results %>% knitr::kable()

# what if we did the average by genre?
genre_bias <- train_set %>% group_by(genres) %>%
  summarize(n = n(), b_g = mean(rating - mu))

predicted_ratings <- test_set %>% left_join(movie_avgs, by='movieId') %>%
  left_join(genre_bias, by='genres') %>% .$b_g + mu

genre_avg_rmse <- RMSE(test_set$rating, predicted_ratings)

# genre_avg_rmse [1.01] 
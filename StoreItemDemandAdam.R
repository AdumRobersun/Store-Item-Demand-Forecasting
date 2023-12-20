#Code for Kaggle Competition: Store Item Demand Forecasting Challenge
#relevant libraries
library(vroom)
library(tidyverse)
library(tidymodels)
library(patchwork)
library(modeltime)
library(timetk)
library(plotly)


train <- vroom("train.csv")
test <- vroom("test.csv")

#Exploratory Data Analysis:
plot1 <- train %>% filter(store == 7, item == 32) %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max = 365)

plot2 <- train %>% filter(store == 1, item == 1) %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max = 30)

plot3 <- train %>% filter(store == 7, item == 7) %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max = 30)

plot4 <- train %>% filter(store == 5, item == 6) %>%
  pull(sales) %>%
  forecast::ggAcf(., lag.max = 30)

plot5 <- (plot1 + plot2) / (plot3 + plot4)


#print plot5
plot5

#explore one store item (store 5 item 15)
numberofstores <- max(train$store)
numberofitems <- max(train$item)

item516 <-train %>% filter(store == 5, item == 16)

time_recipe <- recipe(sales~., data = store_item) %>%
  step_date(date, features = "dow") %>%
  #step_date(date, features = "decimal") %>%
  step_date(date, features = "doy") %>%
  step_range(date_doy, min = 0, max = pi) %>%
  step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy))

knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_engine("kknn") %>%
  set_mode("regression")

knn_wf <- 
  workflow() %>%
  add_model(knn_model) %>%
  add_recipe(time_recipe)

tuning_grid <- grid_regular(neighbors(),
                            levels = 5)

#split into folds
folds <- vfold_cv(store_item, v = 5, repeats = 1)

#run cv
cross_v <-
  knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(smape))

#find best tuning parm values

best_tune <-
  cross_v %>%
  select_best("smape")

collect_metrics(cross_v) %>%
  filter(neighbors == 10) %>%
  pull(mean)

#Random Forest

time_recipe <- recipe(sales~., data = store_item) %>%
  step_date(date, features = "dow") %>%
  step_date(date, features = "doy") %>%
  step_range(date_doy, min = 0, max = pi) %>%
  step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy))

ranfor_mod <- rand_forest(min_n = tune(), mtry = 3) %>%
  set_engine("ranger") %>%
  set_mode("regression")

#create workflow
rf_wf <- 
  workflow() %>%
  add_model(ranfor_mod) %>%
  add_recipe(time_recipe)

#grid for tuning parameters
tuning_grid <- grid_regular(min_n(),
                            levels = 5)

#split into folds
folds <- vfold_cv(store_item, v = 5, repeats = 1)

#run cv

cross_v <-
  rf_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(smape))

#find best tune
best_tune <-
  cross_v %>%
  select_best("smape")

collect_metrics(cross_v) %>%
  filter(min_n == 30) %>%
  pull(mean)






#-----EXPONENTIAL SMOOTHING-----#

#filter to one store and item
store_item1 <-
  train %>% filter(store == 5, item == 16)

store_item_2 <-
  train %>% filter(store == 1, item == 39)

#item 1
# setup up cv

cv_split <- time_series_split(store_item, assess="3 months", cumulative = TRUE)

#set up smoothing model
smoothing_model <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales~date, data = training(cv_split))


## Cross-validate to tune model
cross_v <- modeltime_calibrate(smoothing_model,
                                  new_data = testing(cv_split))

## Visualize CV results
cross_v %>%
modeltime_forecast(
                   new_data = testing(cv_split),
                   actual_data = store_item) %>%
plot_modeltime_forecast(.interactive=TRUE)

p1 <- cross_v %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = store_item) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cross_v %>%
modeltime_accuracy() %>%
table_modeltime_accuracy(.interactive = FALSE)

## Refit to all data then forecast
es_fullfit <- cross_v %>%
modeltime_refit(data = store_item)

es_preds <- es_fullfit %>%
modeltime_forecast(h = "3 months") %>%
rename(date=.index, sales=.value) %>%
select(date, sales) %>%
full_join(., y=test, by="date") %>%
select(id, sales)

es_fullfit %>%
modeltime_forecast(h = "3 months", actual_data = store_item) %>%
plot_modeltime_forecast(.interactive=FALSE)

p2 <- es_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = store_item) %>%
  plot_modeltime_forecast(.interactive=FALSE)

#2nd item
# setup up cv

cv_split <- time_series_split(store_item_2, assess="3 months", cumulative = TRUE)

# set up smoothing model
smoothing_model <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales~date, data = training(cv_split))


## Cross-validate to tune model
cross_v <- modeltime_calibrate(smoothing_model,
                                  new_data = testing(cv_split))

## Visualize CV results
cross_v %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = store_item) %>%
  plot_modeltime_forecast(.interactive=TRUE)

p3 <- cross_v %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = store_item) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cross_v %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

## Refit to all data then forecast
es_fullfit <- cross_v %>%
  modeltime_refit(data = store_item_2)

es_preds <- es_fullfit %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)

es_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = store_item_2) %>%
  plot_modeltime_forecast(.interactive=FALSE)

p4 <- es_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = store_item_2) %>%
  plot_modeltime_forecast(.interactive=FALSE)


subplot(p1,p3,p2,p4, nrows = 2)


### FOR LOOP


nStores <- max(train$store)
nItems <- max(train$item)


# main double-loop to set up store-item pairs

for(s in 1:nStores){
  for(i in 1:nItems){
    storeItemTrain <- train %>%
    filter(store==s, item==i)
    storeItemTest <- test %>%
    filter(store==s, item==i)
    
    ## Fit storeItem models here
    
    ## Predict storeItem sales
    
    ## Save storeItem predictions
    if(s==1 & i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds, preds)
    }
    
  }
}

## SARIMA


## AR - use past to predict future
## MA - use unique part of each day to predict future
## I - differencing
## S - seasonal
## AR uses short-term correlation, MA is longer/smooth


## recipe


cv_split <- time_series_split(store_item, assess="3 months", cumulative = TRUE)
cv_split2 <- time_series_split(store_item_2, assess="3 months", cumulative = TRUE)

store_item_test <-
  test %>% filter(store == 7, item == 32)

store_item_test_2 <-
  test %>% filter(store == 1, item == 3)

arima_recipe <- recipe(sales~., data = store_item) %>%
  step_date(date, features = "dow") %>%
  #step_date(date, features = "decimal") %>%
  step_date(date, features = "doy") %>%
  step_range(date_doy, min = 0, max = pi) %>%
  step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy))

arima_model <- arima_reg(seasonal_period = 365,
                         non_seasonal_ar = 5,
                         non_seasonal_ma = 5,
                         seasonal_ar = 2,
                         seasonal_ma = 2,
                         non_seasonal_differences = 2,
                         seasonal_differences = 2) %>%
  set_engine("auto_arima")


arima_wf <- workflow() %>%
  add_recipe(arima_recipe) %>%
  add_model(arima_model) %>%
  fit(data = training(cv_split))

cross_v <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split))



## Visualize CV results
cross_v %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = store_item) %>%
  plot_modeltime_forecast(.interactive=TRUE)

p1 <- cross_v %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = store_item) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cross_v %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

## Refit to all data then forecast
arima_fullfit <- cross_v %>%
  modeltime_refit(data = store_item)

arima_preds <- arima_fullfit %>%
  modeltime_forecast(new_data = testing(cv_split)) %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)

arima_fullfit %>%
  modeltime_forecast(new_data = store_item_test, actual_data = store_item) %>%
  plot_modeltime_forecast(.interactive=FALSE, .legend_show = FALSE)

p2 <- arima_fullfit %>%
  modeltime_forecast(new_data = store_item_test, actual_data = store_item) %>%
  plot_modeltime_forecast(.interactive=FALSE)


# refit for second combination


arima_recipe <- recipe(sales~., data = store_item_2) %>%
  step_date(date, features = "dow") %>%
  #step_date(date, features = "decimal") %>%
  step_date(date, features = "doy") %>%
  step_range(date_doy, min = 0, max = pi) %>%
  step_mutate(sinDOY = sin(date_doy), cosDOY = cos(date_doy))

arima_model <- arima_reg(seasonal_period = 365,
                         non_seasonal_ar = 5,
                         non_seasonal_ma = 5,
                         seasonal_ar = 2,
                         seasonal_ma = 2,
                         non_seasonal_differences = 2,
                         seasonal_differences = 2) %>%
  set_engine("auto_arima")


arima_wf <- workflow() %>%
  add_recipe(arima_recipe) %>%
  add_model(arima_model) %>%
  fit(data = training(cv_split2))

cross_v <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split2))



## Visualize CV results
cross_v %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = store_item_2) %>%
  plot_modeltime_forecast(.interactive=TRUE)

p3 <- cross_v %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = store_item_2) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cross_v %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

## Refit to all data then forecast
arima_fullfit <- cross_v %>%
  modeltime_refit(data = store_item_2)

arima_preds <- arima_fullfit %>%
  modeltime_forecast(new_data = testing(cv_split2)) %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)

arima_fullfit %>%
  modeltime_forecast(new_data = store_item_test_2, actual_data = store_item_2) %>%
  plot_modeltime_forecast(.interactive=FALSE, .legend_show = FALSE)

p4 <- arima_fullfit %>%
  modeltime_forecast(new_data = store_item_test_2, actual_data = store_item_2) %>%
  plot_modeltime_forecast(.interactive=FALSE)

### create output plots


subplot(p1,p2,p3,p4, nrows = 2)

# cv, pred top row, cv, pred bottom row for 2nd group

# ma short, ar long




### PROPHET MODEL

# store item combo 1
prophet_model <- prophet_reg() %>%
  set_engine("prophet") %>%
  fit(sales ~ date, training(cv_split))

## calibrate the workflow

cross_v <- modeltime_calibrate(prophet_model, new_data = testing(cv_split))



## visualize, evaluate CV accuracy

cross_v %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = store_item) %>%
  plot_modeltime_forecast(.interactive=TRUE)

p1 <- cross_v %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = store_item) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cross_v %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

## refit best model to whole data, predict

prophet_fullfit <- cross_v %>%
  modeltime_refit(data = store_item)

prophet_preds <- prophet_fullfit %>%
  modeltime_forecast(new_data = testing(cv_split)) %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)

p2 <- prophet_fullfit %>%
  modeltime_forecast(new_data = store_item_test, actual_data = store_item) %>%
  plot_modeltime_forecast(.interactive=FALSE, .legend_show = FALSE)

# store item combo 2

prophet_model <- prophet_reg() %>%
  set_engine("prophet") %>%
  fit(sales ~ date, training(cv_split2))

## calibrate the workflow

cross_v <- modeltime_calibrate(prophet_model, new_data = testing(cv_split2))



## visualize, evaluate CV accuracy

cross_v %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = store_item_2) %>%
  plot_modeltime_forecast(.interactive=TRUE)

p3 <- cross_v %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = store_item_2) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cross_v %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

## refit best model to whole data, predict

prophet_fullfit <- cross_v %>%
  modeltime_refit(data = store_item_2)

prophet_preds <- prophet_fullfit %>%
  modeltime_forecast(new_data = testing(cv_split2)) %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)

p4 <- prophet_fullfit %>%
  modeltime_forecast(new_data = store_item_test_2, actual_data = store_item_2) %>%
  plot_modeltime_forecast(.interactive=FALSE, .legend_show = FALSE)


subplot(p1,p3,p2,p4, nrows = 2)


# top row: cv overlaid, bottom row: 3 month forescasts




#### ACTUAL MODEL FITTING


nStores <- max(train$store)
nItems <- max(train$item)


# main double-loop to set up store-item pairs

for(s in 1:nStores){
  for(i in 1:nItems){
    storeItemTrain <- train %>%
      filter(store==s, item==i)
    storeItemTest <- test %>%
      filter(store==s, item==i)
    
    ## Fit storeItem models here
    prophet_model <- prophet_reg() %>%
      set_engine("prophet") %>%
      fit(sales ~ date, storeItemTrain)
    
    ## Predict storeItem sales
    preds <- prophet_model %>% predict(new_data = storeItemTest)
    ## Save storeItem predictions
    if(s==1 & i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds, preds)
    }
    
  }
}

output <- tibble(test$id, all_preds)
output %>% View()



### code to put in notebook - prophet


for(s in 1:nStores){
  for(i in 1:nItems){
    storeItemTrain <- train %>%
      filter(store==s, item==i)
    storeItemTest <- test %>%
      filter(store==s, item==i)
    
    ## Fit storeItem models here
    prophet_model <- prophet_reg() %>%
      set_engine("prophet") %>%
      fit(sales ~ date, storeItemTrain)
    
    ## Predict storeItem sales
    preds <- prophet_model %>% predict(new_data = storeItemTest)
    ## Save storeItem predictions
    if(s==1 & i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds, preds)
    }
    
  }
}

output <- data.frame(id = test$id, sales = all_preds$.pred)
output %>% View()
vroom_write(output, "submission.csv", delim = ",")




#Random Forest Code for Kaggle Notebook
library(tidyverse) 
library(modeltime)
library(tidymodels)
library(vroom)
library(embed)
library(parsnip)
library(bonsai)
library(lightgbm)


item <- vroom::vroom("/kaggle/input/demand-forecasting-kernels-only/train.csv")
itemTest <- vroom::vroom("/kaggle/input/demand-forecasting-kernels-only/test.csv")
n.stores <- max(item$store)
n.items <- max(item$item)

## Define the workflow
item_recipe <- recipe(sales~., data=item) %>%
  step_date(date, features=c("dow", "month", "decimal", "doy", "year")) %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(sales)) %>%
  step_rm(date, item, store) %>%
  step_normalize(all_numeric_predictors())
ranfor_mod <- rand_forest(mtry = 3, min_n = 30) %>%
  set_engine("ranger") %>%
  set_mode("regression")
rf_wf <- workflow() %>%
  add_recipe(item_recipe) %>%
  add_model(ranfor_mod)

# main doube Loop over all store-item combos
for(s in 1:n.stores){
  for(i in 1:n.items){
    
    ## Subset the data
    train <- item %>%
      filter(store==s, item==i)
    test <- itemTest %>%
      filter(store==s, item==i)
    
    ## Fit the data and forecast
    fitted_wf <- rf_wf %>%
      fit(data=train)
    preds <- predict(fitted_wf, new_data=test) %>%
      bind_cols(test) %>%
      rename(sales=.pred) %>%
      select(id, sales)
    
    ## Save the results
    if(s==1 && i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds,
                             preds)
    }
    
  }
}


vroom_write(x=all_preds, "./submission.csv", delim=",")

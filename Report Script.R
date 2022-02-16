if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(janitor)) install.packages("janitor", repos = "http://cran.us.r-project.org")
if(!require(Amelia)) install.packages("Amelia", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(caret)
library(lubridate)
library(knitr)
library(kableExtra)
library(data.table)
library(dplyr)
library(ggplot2)
library(janitor)
library(Amelia)
library(testit)

set.seed(10403)
options(digits = 2)

#load data
claims_df <- read.csv("data/Car_Insurance_Claim.csv")
str(claims_df)

#check duplicates
janitor::get_dupes(dat = claims_df)

#check blanks
Amelia::missmap(claims_df,
                main = "Missing Data Heatmap",
                margins=c(9,5))

#class balance
data.frame(prop.table(table(claims_df$OUTCOME))) %>%
  kbl(.,booktabs = T) %>%
  kable_styling(latex_options = c("striped","hold_position"))

#Charts and EDA
claims_df %>%
  mutate(OUTCOME=as.factor(OUTCOME)) %>%
  ggplot(aes(OUTCOME)) +
  geom_bar(aes(fill=OUTCOME)) +
  ylab("Count of Policies") +
  xlab("Claim Outcome")

claims_df <-
  claims_df %>%
  mutate(OUTCOME=as.factor(OUTCOME))

claims_df %>%
  select_if(is.numeric) %>%
  head() %>%
  kbl(.,booktabs = T) %>%
  kable_styling(latex_options = c("striped","hold_position","scale_down"))

# supress warning about missing data in annual_mileage and credit_score
suppressWarnings(claims_df %>%
                   ggplot(aes(CREDIT_SCORE,colour=OUTCOME,fill=OUTCOME)) +
                   geom_density(alpha=0.5) +
                   xlab("Credit Score") +
                   ylab("Density"))

suppressWarnings(claims_df %>%
                   ggplot(aes(CREDIT_SCORE,colour=INCOME,fill=INCOME)) +
                   geom_density(alpha=0.5) +
                   xlab("Credit Score") +
                   ylab("Density"))

suppressWarnings(claims_df %>%
                   ggplot(aes(ANNUAL_MILEAGE,colour=OUTCOME,fill=OUTCOME)) +
                   geom_density(alpha=0.5) +
                   xlab("Annual Mileage") +
                   ylab("Density"))

suppressWarnings(claims_df %>%
                   mutate(MARRIED=as.factor(MARRIED)) %>%
                   ggplot(aes(ANNUAL_MILEAGE,colour=MARRIED,fill=MARRIED)) +
                   geom_density(alpha=0.5) +
                   xlab("Annual Mileage") +
                   ylab("Density"))

charts_num <- list()
for(i in c("DUIS","SPEEDING_VIOLATIONS","PAST_ACCIDENTS")){
  charts_num[[i]] <-
    claims_df %>%
    ggplot(aes(.data[[i]],fill=OUTCOME))+
    geom_bar(position = 'dodge') +
    ylab("Count of Policies") +
    xlab(i)

}

gridExtra::grid.arrange(grobs = charts_num)

charts <- list()
for(i in c("VEHICLE_TYPE","AGE","GENDER","RACE","DRIVING_EXPERIENCE","EDUCATION","INCOME","MARRIED","CHILDREN","VEHICLE_OWNERSHIP")){
  charts[[i]] <-
    claims_df %>%
    ggplot(aes(.data[[i]],fill=OUTCOME))+
    geom_bar(position = 'dodge') +
    ylab("Count of Policies") +
    xlab(i)

}
gridExtra::grid.arrange(grobs = charts)

data.frame(prop.table(table(na.omit(claims_df)$OUTCOME)))

#fill blanks CREDIT_SCORE
for (i in unique(claims_df$INCOME)) {
  mu <- claims_df %>%
    filter(INCOME == i) %>%
    pull(CREDIT_SCORE) %>%
    mean(.,na.rm=T)

  claims_df$CREDIT_SCORE[
    is.na(claims_df$CREDIT_SCORE) & claims_df$INCOME == i
  ] <- mu
}

for (i in unique(claims_df$MARRIED)) {
  mu <- claims_df %>%
    filter(MARRIED == i) %>%
    pull(ANNUAL_MILEAGE) %>%
    mean(.,na.rm=T)

  claims_df$ANNUAL_MILEAGE[
    is.na(claims_df$ANNUAL_MILEAGE)& claims_df$MARRIED == i
  ] <- mu
}


#drop ID
claims_df <-
  claims_df %>%
  select(-ID)

#Train/Test splitting
test_index <- createDataPartition(y = claims_df$OUTCOME, times = 1, p = 0.2, list = FALSE)
train <- claims_df[-test_index,]
test <- claims_df[test_index,]

#class balance of train/test sets
data.frame(prop.table(table(na.omit(train)$OUTCOME)))
data.frame(prop.table(table(na.omit(test)$OUTCOME)))

#GLM ----
fit_glm <-
  caret::train(OUTCOME ~ .,
               data=train,
               method="glm")

#LDA ----
fit_lda <-
  caret::train(OUTCOME ~ .,
               data=train,
               method="lda")

#QDA ----
fit_qda <-
  caret::train(OUTCOME ~ .,
               data=train,
               method="qda")

#kNN ----
fit_knn <-
  caret::train(OUTCOME ~ .,
               data=train,
               method="knn",
               tuneGrid=data.frame(k=seq(1,16,2)),
               trControl = trainControl(method = "cv",
                                        number = 10))

#Classification Tree ----
cp_grid <- data.frame(cp=seq(0, 0.03, 0.002))
fit_ct <- caret::train(OUTCOME ~ .,
                       data = train,
                       method="rpart",
                       tuneGrid = cp_grid)

#Random Forest ----
mtry_grid = data.frame(mtry = seq(1,7,2))
fit_rf <- caret::train(OUTCOME ~ .,
                       data = train,
                       method='rf',
                       tuneGrid=mtry_grid,
                       ntree=100)

#treebag ----
fit_tb <-
  caret::train(OUTCOME ~ .,
               data=train,
               method="treebag")

#C4.5 ----
fit_c4.5 <- caret::train(OUTCOME ~ .,
                         data = train,
                         method='J48')

#C5 ----
fit_c5 <- caret::train(OUTCOME ~ .,
                       data = train,
                       method='C5.0')
#FDA ----
fit_fda <- caret::train(OUTCOME ~ .,
                        data = train,
                        method='fda')

#XGBtree ----
fit_xgb <-
  caret::train(OUTCOME ~ .,
               data=train,
               method='xgbTree')

# Model Summary -----
models <-
  c("GLM", "LDA", "QDA", "kNN","Classification Tree", "Random forest",
    "Treebag","C4.5","C5","FDA","xgbTree")

model_name <-
  list(fit_glm, fit_lda, fit_qda, fit_knn,fit_ct, fit_rf,
       fit_tb,fit_c4.5,fit_c5,fit_fda,fit_xgb)

accuracy <- list()
sensitivity <- list()
specificity <- list()
index <-0
for (i in model_name){
  index <- index + 1
  accuracy[[index]] <-confusionMatrix(predict(i, test), test$OUTCOME)$overall["Accuracy"]
  sensitivity[[index]] <-confusionMatrix(predict(i, test), test$OUTCOME)$byClass["Sensitivity"]
  specificity[[index]] <-confusionMatrix(predict(i, test), test$OUTCOME)$byClass["Specificity"]
}

model_summary <-
  data.frame(Model = models,
             Accuracy = unlist(accuracy),
             Specificity = unlist(specificity),
             Sensitivity = unlist(sensitivity))

model_summary <-
  model_summary %>%
  mutate(Mean = rowMeans(select(model_summary, is.numeric), na.rm = TRUE)) %>%
  arrange(desc(Mean)) %>%
  select(-Mean)

model_summary

confusion_matrix_plot <- function(model){
  table <- data.frame(confusionMatrix(predict(model, test), test$OUTCOME)$table)

  plotTable <- table %>%
    mutate(Classification = ifelse(table$Prediction == table$Reference, "Correct", "Incorrect")) %>%
    group_by(Reference) %>%
    mutate(prop = Freq/sum(Freq))

  ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = Classification, alpha = prop)) +
    geom_tile() +
    geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
    scale_fill_manual(values = c(Correct = "green", Incorrect = "red")) +
    theme_bw() +
    xlim(rev(levels(table$Reference)))
}

confusion_matrix_plot(fit_xgb)


#Tuning xgbTree
tune_grid <- expand.grid(
  nrounds = seq(from = 100, to = 1000, by = 50),
  eta = c(0.025, 0.05, 0.1, 0.5),
  max_depth = c(2, 3, 4, 5, 10),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

tune_control <- caret::trainControl(
  method = "cv", # cross-validation
  number = 3, # with n folds
  verboseIter = FALSE, # no training log
  allowParallel = TRUE # FALSE for reproducible results
)

xgb_tune <- caret::train(
  OUTCOME ~ .,
  data=train,
  trControl = tune_control,
  tuneGrid = tune_grid,
  method = "xgbTree",
  verbose = TRUE
)

xgb_tune$bestTune

tuneplot <- function(x, probs = .95) {
  ggplot(x) +
    coord_cartesian(ylim = c(min(x$results$Accuracy),
                             quantile(x$results$Accuracy,
                                      probs = probs))
    ) +
    theme_bw()
}

tuneplot(xgb_tune)

tune_grid2 <- expand.grid(
  nrounds = seq(from = 50, to = 1000, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = ifelse(xgb_tune$bestTune$max_depth == 2,
                     c(xgb_tune$bestTune$max_depth:4),
                     xgb_tune$bestTune$max_depth - 1:xgb_tune$bestTune$max_depth + 1),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3),
  subsample = 1
)

xgb_tune2 <- caret::train(
  OUTCOME ~ .,
  data=train,
  trControl = tune_control,
  tuneGrid = tune_grid2,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune2)

xgb_tune2$bestTune

tune_grid3 <- expand.grid(
  nrounds = seq(from = 50, to = 1000, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = 0,
  colsample_bytree = c(0.4, 0.6, 0.8, 1.0),
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = c(0.5, 0.75, 1.0)
)

xgb_tune3 <- caret::train(
  OUTCOME ~ .,
  data=train,
  trControl = tune_control,
  tuneGrid = tune_grid3,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune3, probs = .90)

xgb_tune3$bestTune

tune_grid4 <- expand.grid(
  nrounds = seq(from = 100, to = 1000, by = 50),
  eta = xgb_tune$bestTune$eta,
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = c(0, 0.05, 0.1, 0.5, 0.8, 1.0),
  colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = xgb_tune3$bestTune$subsample
)

xgb_tune4 <- caret::train(
  OUTCOME ~ .,
  data=train,
  trControl = tune_control,
  tuneGrid = tune_grid4,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune4,0.975)

xgb_tune4$bestTune

tune_grid5 <- expand.grid(
  nrounds = seq(from = 100, to = 10000, by = 100),
  eta = c(0.01, 0.015, 0.025, 0.05, 0.1),
  max_depth = xgb_tune2$bestTune$max_depth,
  gamma = xgb_tune4$bestTune$gamma,
  colsample_bytree = xgb_tune3$bestTune$colsample_bytree,
  min_child_weight = xgb_tune2$bestTune$min_child_weight,
  subsample = xgb_tune3$bestTune$subsample
)


xgb_tune5 <- caret::train(
  OUTCOME ~ .,
  data=train,
  trControl = tune_control,
  tuneGrid = tune_grid5,
  method = "xgbTree",
  verbose = TRUE
)

tuneplot(xgb_tune5)

xgb_tune5$bestTune

#Final tuning and fitting model
final_grid <- expand.grid(
  nrounds = xgb_tune5$bestTune$nrounds,
  eta = xgb_tune5$bestTune$eta,
  max_depth = xgb_tune5$bestTune$max_depth,
  gamma = xgb_tune5$bestTune$gamma,
  colsample_bytree = xgb_tune5$bestTune$colsample_bytree,
  min_child_weight = xgb_tune5$bestTune$min_child_weight,
  subsample = xgb_tune5$bestTune$subsample
)

train_control <- caret::trainControl(
  method = "none",
  verboseIter = FALSE,
  allowParallel = TRUE
)

xgb_model <- caret::train(
  OUTCOME ~ .,
  data=train,
  trControl = train_control,
  tuneGrid = final_grid,
  method = "xgbTree",
  verbose = TRUE
)

model_summary %>%
  rbind(tibble(Model="Tuned xgbTree",
               Accuracy =confusionMatrix(predict(xgb_model, test), test$OUTCOME)$overall["Accuracy"],
               Sensitivity = confusionMatrix(predict(xgb_model, test), test$OUTCOME)$byClass["Sensitivity"],
               Specificity = confusionMatrix(predict(xgb_model, test), test$OUTCOME)$byClass["Specificity"])
  )

model_summary <-
  model_summary %>%
  mutate(Mean = rowMeans(select(model_summary, is.numeric), na.rm = TRUE)) %>%
  arrange(desc(Mean)) %>%
  select(-Mean)

model_summary

confusion_matrix_plot(xgb_model)

#Ensemble modeling
y_models <-
  bind_cols(
    y_glm = predict(fit_glm,test),
    y_lda = predict(fit_lda,test),
    y_fda = predict(fit_fda,test),
    y_qda = predict(fit_qda,test),
    y_c5 = predict(fit_c5,test),
    y_ct = predict(fit_ct,test),
    y_xgb = predict(xgb_model,test)
  )

ensemble_pred <-
  y_models %>%
  mutate_all(as.numeric) %>%
  rowMeans()

ensemble_pred <-
  ifelse(ensemble_pred > 1.5, 1,0)

model_summary <-
  model_summary %>%
  rbind(tibble(Model="Ensemble Model",
               Accuracy =confusionMatrix(factor(ensemble_pred), test$OUTCOME)$overall["Accuracy"],
               Sensitivity = confusionMatrix(factor(ensemble_pred), test$OUTCOME)$byClass["Sensitivity"],
               Specificity = confusionMatrix(factor(ensemble_pred), test$OUTCOME)$byClass["Specificity"])
  )

model_summary <-
  model_summary %>%
  mutate(Mean = rowMeans(select(model_summary, is.numeric), na.rm = TRUE)) %>%
  arrange(desc(Mean)) %>%
  select(-Mean)

model_summary

table <- data.frame(confusionMatrix(factor(ensemble_pred),test$OUTCOME)$table)

plotTable <- table %>%
  mutate(Classification = ifelse(table$Prediction == table$Reference, "Correct", "Incorrect")) %>%
  group_by(Reference) %>%
  mutate(prop = Freq/sum(Freq))

ggplot(data = plotTable, mapping = aes(x = Reference, y = Prediction, fill = Classification, alpha = prop)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = .5, fontface  = "bold", alpha = 1) +
  scale_fill_manual(values = c(Correct = "green", Incorrect = "red")) +
  theme_bw() +
  xlim(rev(levels(table$Reference)))

# set seed ----
set.seed(10403)

#Import data -----
claims_df <- read.csv("data/Car_Insurance_Claim.csv")

#fix missing blanks in CREDIT_SCORE - with average as a function of INCOME ----
for (i in unique(claims_df$INCOME)) {
  mu <- claims_df %>%
    filter(INCOME == i) %>%
    pull(CREDIT_SCORE) %>%
    mean(.,na.rm=T)

  claims_df$CREDIT_SCORE[
    is.na(claims_df$CREDIT_SCORE) & claims_df$INCOME == i
  ] <- mu
}

#fix missing blanks in ANNUAL_MILEAGE -----
for (i in unique(claims_df$MARRIED)) {
  mu <- claims_df %>%
    filter(MARRIED == i) %>%
    pull(CREDIT_SCORE) %>%
    mean(.,na.rm=T)

  claims_df$ANNUAL_MILEAGE[
    is.na(claims_df$ANNUAL_MILEAGE)& claims_df$MARRIED == i
    ] <- mu

}

#set Target Variable as factor ----
claims_df <-
  claims_df %>%
  mutate(OUTCOME=as.factor(OUTCOME),
         POSTAL_CODE=as.character(POSTAL_CODE))

#Encode labels ----
for (i in colnames(claims_df %>% select_if(is.character))){
lbl = LabelEncoder$new()
lbl$fit(claims_df[[i]])
claims_df[[i]] <- lbl$fit_transform(claims_df[[i]])
}

#Scale data ----
claims_scaled <-
  claims_df %>% select(-OUTCOME,-ID)

# normalize <- function(x) {
#   return ((x - min(x)) / (max(x) - min(x)))
# }
#
# for (i in colnames(claims_scaled)){
#   claims_scaled[[i]] <-
#     normalize(claims_scaled[[i]])
# }

claims_scaled <-
  scale(claims_scaled)

claims_scaled <-
data.frame(claims_scaled)

#add OUTCOME ----
claims_scaled <-
  claims_scaled %>%
  bind_cols(OUTCOME=claims_df$OUTCOME)

#Test train Split----
test_index <- createDataPartition(y = claims_scaled$OUTCOME,
                                  times = 1, p = 0.2, list = FALSE)
train <- claims_scaled[-test_index,]
test <- claims_scaled[test_index,]


#Balance data with Random Over Sampling
#train.rose <- ROSE(OUTCOME ~ ., data=train, seed=10403)$data

#Model testing ----
#C5.0 ----
fit_c5 <- caret::train(OUTCOME ~ .,
                       data = train,
                       method='C5.0')
fit_c5
y_hat <- predict(fit_c5,test)
confusionMatrix(y_hat,reference = test$OUTCOME)


#Random forest -----
mtry_grid = data.frame(mtry = seq(1,7,2))
fit_rf <- caret::train(OUTCOME ~ .,
                       data = train,
                       method='rf',
                       tuneGrid=mtry_grid,
                       ntree=100)

fit_rf
y_hat <- predict(fit_rf,test)
confusionMatrix(y_hat,test$OUTCOME)

#FDA ----
fit_fda <- caret::train(OUTCOME ~ .,
                        data = train,
                        method='fda')
fit_fda
y_hat <- predict(fit_fda,test)
confusionMatrix(y_hat,reference = test$OUTCOME)

#Classification Tree ----
cp_grid <- data.frame(cp=seq(0, 0.05, 0.002))
fit_ct <- caret::train(OUTCOME ~ .,
                       data = train,
                       method="rpart",
                       tuneGrid = cp_grid)

fit_ct
y_hat <- predict(fit_ct,test)
confusionMatrix(y_hat,reference = test$OUTCOME)

#kNN ----
fit_knn <-
  caret::train(OUTCOME ~ .,
               data=train,
               method="knn",
               tuneGrid=data.frame(k=seq(1,51,2)),
               trControl = trainControl(method = "cv",
                                        number = 5))
fit_knn
y_hat <- predict(fit_knn,test)
confusionMatrix(y_hat,reference = test$OUTCOME)

#GLM ----
fit_glm <-
  caret::train(OUTCOME ~ .,
               data=train,
               method="glm")

y_hat <- predict(fit_glm,test)
confusionMatrix(y_hat,reference = test$OUTCOME)

#QDA ----
fit_qda <-
  caret::train(OUTCOME ~ .,
               data=train,
               method="qda")

y_hat <- predict(fit_qda,test)
confusionMatrix(y_hat,reference = test$OUTCOME)

#LDA ----
fit_lda <-
  caret::train(OUTCOME ~ .,
               data=train,
               method="lda")

y_hat <- predict(fit_lda,test)
confusionMatrix(y_hat,reference = test$OUTCOME)

#Treebag ----
fit_tb <-
  caret::train(OUTCOME ~ .,
               data=train,
               method="treebag")

y_hat <- predict(fit_tb,test)
confusionMatrix(y_hat,reference = test$OUTCOME)


#C4.5 ----
fit_c4.5 <- caret::train(OUTCOME ~ .,
                         data = train,
                         method='J48')
fit_c4.5
y_hat <- predict(fit_c4.5,test)
confusionMatrix(y_hat,reference = test$OUTCOME)


# Model Summary -----
models <-
  c("GLM", "LDA", "QDA", "kNN","Classification Tree", "Random forest",
    "Treebag","C4.5","C5","FDA")
accuracy <-
  c(confusionMatrix(predict(fit_glm, test), test$OUTCOME)$overall["Accuracy"],
    confusionMatrix(predict(fit_lda, test), test$OUTCOME)$overall["Accuracy"],
    confusionMatrix(predict(fit_qda, test), test$OUTCOME)$overall["Accuracy"],
    confusionMatrix(predict(fit_knn, test), test$OUTCOME)$overall["Accuracy"],
    confusionMatrix(predict(fit_ct, test), test$OUTCOME)$overall["Accuracy"],
    confusionMatrix(predict(fit_rf, test), test$OUTCOME)$overall["Accuracy"],
    confusionMatrix(predict(fit_tb, test), test$OUTCOME)$overall["Accuracy"],
    confusionMatrix(predict(fit_c4.5, test), test$OUTCOME)$overall["Accuracy"],
    confusionMatrix(predict(fit_c5, test), test$OUTCOME)$overall["Accuracy"],
    confusionMatrix(predict(fit_fda, test), test$OUTCOME)$overall["Accuracy"]
  )

sensitivity <-
  c(confusionMatrix(predict(fit_glm, test), test$OUTCOME)$byClass['Sensitivity'],
    confusionMatrix(predict(fit_lda, test), test$OUTCOME)$byClass['Sensitivity'],
    confusionMatrix(predict(fit_qda, test), test$OUTCOME)$byClass["Sensitivity"],
    confusionMatrix(predict(fit_knn, test), test$OUTCOME)$byClass["Sensitivity"],
    confusionMatrix(predict(fit_ct, test), test$OUTCOME)$byClass["Sensitivity"],
    confusionMatrix(predict(fit_rf, test), test$OUTCOME)$byClass["Sensitivity"],
    confusionMatrix(predict(fit_tb, test), test$OUTCOME)$byClass["Sensitivity"],
    confusionMatrix(predict(fit_c4.5, test), test$OUTCOME)$byClass["Sensitivity"],
    confusionMatrix(predict(fit_c5, test), test$OUTCOME)$byClass["Sensitivity"],
    confusionMatrix(predict(fit_fda, test), test$OUTCOME)$byClass["Sensitivity"]
  )

specificity <-
  c(confusionMatrix(predict(fit_glm, test), test$OUTCOME)$byClass["Specificity"],
    confusionMatrix(predict(fit_lda, test), test$OUTCOME)$byClass["Specificity"],
    confusionMatrix(predict(fit_qda, test), test$OUTCOME)$byClass["Specificity"],
    confusionMatrix(predict(fit_knn, test), test$OUTCOME)$byClass["Specificity"],
    confusionMatrix(predict(fit_ct, test), test$OUTCOME)$byClass["Specificity"],
    confusionMatrix(predict(fit_rf, test), test$OUTCOME)$byClass["Specificity"],
    confusionMatrix(predict(fit_tb, test), test$OUTCOME)$byClass["Specificity"],
    confusionMatrix(predict(fit_c4.5, test), test$OUTCOME)$byClass["Specificity"],
    confusionMatrix(predict(fit_c5, test), test$OUTCOME)$byClass["Specificity"],
    confusionMatrix(predict(fit_fda, test), test$OUTCOME)$byClass["Specificity"]
  )

Models_scaled <-
  data.frame(Model = models,
             Accuracy = accuracy,
             Specificity = specificity,
             Sensitivity = sensitivity) %>%
  arrange(desc(Accuracy))

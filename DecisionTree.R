# Decsion Trees
# Read the input data
data=read.csv(file.choose(),stringsAsFactors=FALSE)

# Assign the training and testing set in the ratio of 80:20
data_train=data[1:118793,]
data_test=data[118794:148491,]

# Show the split of training and testing data
prop.table(table(data_train$labels))
prop.table(table(data_test$labels))

# Load the rpart and caret library
library(rpart)
library(caret)

# 10-fold cross-validation technique
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# Train the decision tree classifier
dtree_fit <- train(labels ~., data = data_train, method = "rpart",
                   parms = list(split = "information"),
                   trControl=trctrl,
                   tuneLength = 10)

# Look at basic information about the model
dtree_fit

# Predict engagement of tweets
test_pred <- predict(dtree_fit, newdata = data_test)

# Print the confusion matrix
confusionMatrix(test_pred, data_test$labels)

# Train the decision tree classifier with criterion as gini index
dtree_fit_gini <- train(labels ~., data = data_train, method = "rpart",
                        parms = list(split = "gini"),
                        trControl=trctrl,
                        tuneLength = 10)

# Look at basic information about the model
dtree_fit_gini

# Predict engagement of tweets
test_pred_gini <- predict(dtree_fit_gini, newdata = data_test)

# Print the confusion matrix
confusionMatrix(test_pred_gini, data_test$labels)

#Random Forest
require(randomForest)
require(MASS)
fit <- randomForest(as.factor(labels) ~ .,
                    data=data_train, 
                    importance=TRUE)

#importance(fit)

# Predict the engagement of tweets
Prediction <- predict(fit, data_test)
table(predict(fit),data_train$labels)

# Print the confusion matrix
confusionMatrix(Prediction, data_test$labels)

# Boosting
library(gbm)
tweets_boost <- gbm(labels ~ . ,data = data_train,distribution = "gaussian",n.trees = 10000,
                 shrinkage = 0.01, interaction.depth = 4)
tweets_boost
tweets_boost_pred <- predict(tweets_boost,data_test,n.trees = 10000)
confusionMatrix(tweets_boost_pred, data_test$labels)

# Print accuray for the model while predicting with only 2 labels
categories <- data_test[,"labels"]
library(cvAUC)
cvAUC::AUC(predictions = tweets_boost_pred, labels = categories)

# Plot ROCR Curve
library(ROCR)

d = data
mylogit <- glm(d$labels~d$number_of_hashtags+d$number_of_mentions+d$number_of_media+d$liwc_achievement+d$liwc_anger+d$liwc_emotional+d$liwc_envy+d$liwc_hate+d$liwc_irritability+d$liwc_joy+d$liwc_love+d$liwc_rage+d$liwc_sympathy+d$liwc_family+d$liwc_fashion+d$liwc_health+d$liwc_exercise+d$liwc_music+d$liwc_politics+d$liwc_government+d$liwc_positive_emotion+d$liwc_negative_emotion+d$liwc_sports+d$liwc_traveling+d$liwc_vacation, family = "binomial")
summary(mylogit)

d$score <-predict.glm(mylogit, type="response" )
pred<-prediction(d$score,d$labels)
perf<-performance(pred,"tpr", "fpr")
plot(perf)
abline(a=0,b=1)

# Compute sensitivity, specificity and cutoff
opt.cut = function(perf, pred) {	
  cut.ind = mapply(FUN = function(x,y,p) {
    d = (x-0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}
print(opt.cut(perf,pred))

# Accuracy Plot
acc.perf = performance(pred, measure = "acc")
plot(acc.perf)

## Maximum Accuracy
ind = which.max(slot(acc.perf, 'y.values')[[1]])
acc = slot(acc.perf, 'y.values')[[1]][ind]
cutoff = slot(acc.perf, 'x.values')[[1]][ind]
print(c(accuracy = acc, cutoff = cutoff))

# Area Under The Curve
auc.perf = performance(pred, measure = "auc")
auc.perf@y.values

# Precision and Recall
perf1 <- performance(pred, "prec", "rec")
plot(perf1)

precision1 <- table(tweets_boost_pred, data_test$labels)[1,1]/sum(table(tweets_boost_pred, data_test$labels)[1,1:2])
precision1

recall1 <- table(tweets_boost_pred, data_test$labels)[1,1]/sum(table(tweets_boost_pred, data_test$labels)[1:2,1])
recall1

# F-1 Measure
f1_measure1 <- 2 * precision1 * recall1 / (precision1 + recall1)
f1_measure1

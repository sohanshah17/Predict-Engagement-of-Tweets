# Naive Bayes
# Read the input data
data=read.csv(file.choose(),stringsAsFactors=FALSE)

# Assign the training and testing set in the ratio of 80:20
data_train=data[1:118782,]
data_test=data[118783:148477,]

# Show the split of training and testing data
prop.table(table(data_train$labels))
prop.table(table(data_test$labels))

library(e1071)

# Train the classifier
nb_classifier=naiveBayes(as.factor(data_train$labels)~.,data=data_train,laplace = 1)
nb_classifier_pred=predict(nb_classifier,data_test[,-26])

# Predict engagement of tweets
table(nb_classifier_pred, data_test[,26], dnn=list('predicted','actual'))

library(caret)

# Print confusion matrix and accuracy
conf.mat <- confusionMatrix(nb_classifier_pred, data_test$labels)
conf.mat
conf.mat$byClass
conf.mat$overall
conf.mat$overall['Accuracy']

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

precision1 <- table(nb_classifier_pred, data_test$labels)[1,1]/sum(table(nb_classifier_pred, data_test$labels)[1,1:2])
precision1

recall1 <- table(nb_classifier_pred, data_test$labels)[1,1]/sum(table(nb_classifier_pred, data_test$labels)[1:2,1])
recall1

# F-1 Measure
f1_measure1 <- 2 * precision1 * recall1 / (precision1 + recall1)
f1_measure1

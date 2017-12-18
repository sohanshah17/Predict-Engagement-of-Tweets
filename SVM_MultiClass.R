# Multiclass SVM
# Read the input data
tweets <- read.csv(file.choose())

# Assign the training and testing set in the ratio of 80:20
tweets_train <- tweets[1:118793,]
tweets_test <- tweets[118794:148491,]

# Load the kernlab library
library(kernlab)

# Train the SVM model
tweet_classifier <- ksvm(labels~., data = tweets_train, kernel = "vanilladot")

# Look at basic information about the model
tweet_classifier

# Evaluate model performance
tweet_predictions <- predict(tweet_classifier, tweets_test)
head(tweet_predictions)

# Display the confusion matrix
table(tweet_predictions, tweets_test$labels)

# Calculate overall accuracy
agreement <- tweet_predictions == tweets_test$labels
table(agreement)
prop.table(table(agreement))


# RBF Kernel
tweet_classifier_rbf <- ksvm(labels~., data = tweets_train, kernel = "rbfdot")
tweet_predictions_rbf <- predict(tweet_classifier_rbf, tweets_test)
table(tweet_predictions_rbf, tweets_test$labels)
agreement <- tweet_predictions_rbf == tweets_test$labels
table(agreement)
prop.table(table(agreement))

# Image consulting through data science
This project presents a machine learning model that predicts the engagement of a tweet.

Users and organizations care not only about the reach and impression of their tweets but also about the engagement for each tweet. It is of no value if the tweet has a high “potential” reach or has been viewed by many people, if nobody has interacted with it. Engagement is the metric that we use to measure such an interaction.

Engagement Score = (Number of Retweets + Number of Favorites) / (Number of Followers)

This model uses linguisting features of the tweet to predict engagement. As a result, engagement can be predicted even before the tweet is posted on twitter.

# Data Set
150,000 tweets extracted from 60 celebrity Twitter accounts.
Each tweet was labeled either "High" or "Low" depending on the engagement score calculated.
25 linguisting features were used as variables to make the prediction - achievement, anger, emotional, envy, hate, irritability, joy, love, rage, sympathy, family, fashion, health, exercise, music, politics, government, positive emotion, negative emotion, sports, traveling and vacation.

# Approach
Several classification algorithms were used to make the prediction. 80% of the data set was used for training and the remaining 20% for testing. Decision Trees gave the highest accuracy of 73.6%.

# Conclusion
The engagement of a tweet can be predicted with an accuracy of 73.6% using just linguistic features from the tweet. This means that the prediction can be made even before the tweet is posted on twitter, giving the end-user the opportunity to modify tweet content to receive maximum engagement.

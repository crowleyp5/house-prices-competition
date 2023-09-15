# house-prices-competition
96th percentile score on Kaggle's "House Prices - Advanced Regression Techniques" competition

This is the code I used to generate house sale price predictions. Kaggle scores the predictions based on the root mean squared error. The scores are put on a leaderboard, and the submissions that are more than four months old are removed from the leaderboard to keep the competition constantly fresh. My scores were about 96th percentile.

I used this competition to practice hyperparamter optimization and ensemble learning methods. I achieved this through a series of small grid searches to find optimal parameters and by leveraging mutliple ensemble learning models such as xgboost and lightgbm. I added lasso regression to the ensemble to help focus the predictions on the features that were most correlated with sale price. There were quite a few parameters to optimize, so I played around with some hyperparameters manually rather than running a computationally expensive grid search for everything.

Kaggle already split the dataset into a train and test set, but the sale prices were removed from the test set to prevent cheating. This meant that I had to split up the training data even more for cross validation during the hyperparameter optimization process.

Overall, this competition was a great learning experience.

Proof of score and percentile:

![crowleyp5Kaggle1](https://github.com/crowleyp5/house-prices-competition/assets/108026566/b47233f3-094d-4b9a-8b4b-02be7904451c)
![crowleyp5Kaggle2](https://github.com/crowleyp5/house-prices-competition/assets/108026566/e2eca27b-fa58-47a6-8c89-5acb3dd74657)

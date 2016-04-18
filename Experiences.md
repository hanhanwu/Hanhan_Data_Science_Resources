This file contains experiences, suggestions when doing data analysis

PREDICTIVE MODELS

-- RAMDOM FORESTS

1. Random Forest is a powerful algorithm which holistically takes care of missing values, 
outliers and other non-linearities in the data set. It’s simply a collection of classification trees, 
hence the name ‘forest’.
2. Random forest has a feature of presenting the important variables.
3. Try one hot encoding and label encoding for random forest model.
4. Parameters Tuning will help.
5. Use Gradient Boosting with Random Forests
6. Build an ensemble of these models: http://www.analyticsvidhya.com/blog/2015/09/questions-ensemble-modeling/
7. In R library(randomForest), method = “parRF”. This is parallel random forest. 
This is parallel implementation of random forest. 
This package causes your local machine to take less time in random forest computation. 
Alternatively, you can also use method = “rf” as a standard random forest function.
8. Using Random Forests in R, Python: 
http://www.analyticsvidhya.com/blog/2015/09/random-forest-algorithm-multiple-challenges/


-- Linear Regression

1. Linear Regression takes following assumptions:
There exists a linear relationship between response and predictor variables
The predictor (independent) variables are not correlated with each other. Presence of collinearity leads to a phenomenon known as multicollinearity.
The error terms are uncorrelated. Otherwise, it will lead to autocorrelation.
Error terms must have constant variance. Non-constant variance leads to heteroskedasticity.

This file contains experiences, suggestions when doing data analysis

PREDICTIVE MODELS

-- Random Forests

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
9. Tuning Random Forests parameters (Python Scikit-Learn): http://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/
10. Be careful that random forests have a tendency to bias towards variables that have more number of distinct values. Foe example, it favors numeric variables over binary/categorical values.


-- Linear Regression

1. Linear Regression takes following assumptions:
 * There exists a linear relationship between response (dependent) and predictor (independent) variables
 * The predictor (independent) variables are not correlated with each other. Presence of collinearity leads to a phenomenon known as multicollinearity.
 * The error terms are uncorrelated. Otherwise, it will lead to autocorrelation.
 * Error terms must have constant variance. Non-constant variance leads to heteroskedasticity.


Note: Linear Regression is very sensitive to Outliers. It can terribly affect the regression line and eventually the forecasted values.

2. There are two common algorithms to find the right coefficients for minimum sum of squared errors, first one is Ordinary Least Sqaure (OLS, used in python library sklearn) and other one is gradient descent. 
* OLS: http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
* <b>Performance Evaluation Metrics</b> for Linear Regression:
 * SSE - minimum sum of squared errors (SSE), but it highly sensitive to the number of data points.
 * R-Square: How much the change in output variable (y) is explained by the change in input variable(x). Its value is between 0 and 1, 0 indicates that the model explains NIL variability in the response data around its mean, 1 indicates that the model explains full variability in the response data around its mean. R² has less variation in score compare to SSE. One disadvantage of R-squared is that it can only increase as predictors are added to the regression model. This increase is artificial when predictors are not actually improving the model’s fit.
 * Adjusted R-Square: To cure the disadvantage in R-Square.  Adjusted R-squared will decrease as predictors are added if the increase in model fit does not make up for the loss of degrees of freedom. Likewise, it will increase as predictors are added if the increase in model fit is worthwhile. Adjusted R-squared should always be used with models with more than one predictor variable.
* Multi-Variate Regression - 1+ predictors. Things get much more complicated when your multiple independent variables are related to with each other. This phenomenon is known as Multicollinearity. This is undesirable.  To avoid such situation, it is advisable to look for Variance Inflation Factor (VIF). For no multicollinearity, VIF should be ( VIF < 2). In case of high VIF, look for correlation table to find highly correlated variables and drop one of correlated ones.
* Along with multi-collinearity, regression suffers from Autocorrelation, Heteroskedasticity.
* Multicollinearity can increase the variance of the coefficient estimates and make the estimates very sensitive to minor changes in the model. The result is that the coefficient estimates are unstable
* In case of multiple independent variables, we can go with forward selection, backward elimination and step wise approach for selection of most significant independent variables.
* Linear Regression with basic R, Python code: http://www.analyticsvidhya.com/blog/2015/10/regression-python-beginners/


-- 7 Types of Regression (Advanced Regression Techniques)

http://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/

 * Regression analysis estimates the relationship between two or more variables.
 * It indicates the significant relationships between dependent variable and independent variable.
 * It indicates the strength of impact of multiple independent variables on a dependent variable.
 * Types of regression mostly depend on: number of independent variables, type of dependent variables and shape of regression line.


-- Logistic Regression

 * It is widely used for classification problems
 * Logistic regression doesn’t require linear relationship between dependent and independent variables.  It can handle various types of relationships because it applies a non-linear log transformation to the predicted odds ratio
 * To avoid over fitting and under fitting, we should include all significant variables. A good approach to ensure this practice is to use a <b>step wise method</b> to estimate the logistic regression
 * It requires large sample sizes because maximum likelihood estimates are less powerful at low sample sizes than ordinary least square
 * The independent variables should not be correlated with each other i.e. no multi collinearity.  However, we have the options to include interaction effects of categorical variables in the analysis and in the model.
 * If the values of dependent variable is ordinal, then it is called as Ordinal logistic regression 
 * If dependent variable is multi class then it is known as Multinomial Logistic regression.


-- Other Commonly Used Regression

* <b>Polynomial Regression</> - A regression equation is a polynomial regression equation if the power of independent variable is more than 1. The aim of this modeling technique is to maximize the prediction power with minimum number of predictor variables.
* <b>Stepwise Regression</b> - This form of regression is used when we deal with multiple independent variables. In this technique, the selection of independent variables is done with the help of an automatic process, which involves no human intervention.
 * Stepwise regression basically fits the regression model by adding/dropping co-variates one at a time based on a specified criterion.
 * Standard stepwise regression does two things. It adds and removes predictors as needed for each step.
 * Forward selection starts with most significant predictor in the model and adds variable for each step.
 * Backward elimination starts with all predictors in the model and removes the least significant variable for each step.
* <b>Ridge Regression</b> - a technique used when the data suffers from multicollinearity ( independent variables are highly correlated).
 * In multicollinearity, even though the least squares estimates (OLS) are unbiased, their variances are large which deviates the observed value far from the true value. By adding a degree of bias to the regression estimates, ridge regression reduces the standard errors.
 * In a linear equation, prediction errors can be decomposed into two sub components. First is due to the biased and second is due to the variance.
 * Ridge regression solves the multicollinearity problem through shrinkage parameter λ (lambda).
 * It shrinks the value of coefficients but doesn’t reaches zero, which suggests no feature selection feature.
 * This is a regularization method and uses l2 regularization.
* <b>Lasso Regression</b> - Similar to Ridge Regression, Lasso (Least Absolute Shrinkage and Selection Operator) also penalizes the absolute size of the regression coefficients. In addition, it is capable of reducing the variability and improving the accuracy of linear regression models.
 * Lasso regression differs from ridge regression in a way that it uses absolute values in the penalty function, instead of squares. This leads to penalizing (or equivalently constraining the sum of the absolute values of the estimates) values which causes some of the parameter estimates to turn out exactly zero, which certainly helps in feature selection.
 * This is a regularization method and uses l1 regularization
 * If group of predictors are highly correlated, lasso picks only one of them and shrinks the others to zero
* <b>ElasticNet Regression</b> - ElasticNet is hybrid of Lasso and Ridge Regression techniques. It is trained with L1 and L2 prior as regularizer.
 * Elastic-net is useful when there are multiple features which are correlated. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both. It encourages group effect in case of highly correlated variables
 * A practical advantage of trading-off between Lasso and Ridge is that, it allows Elastic-Net to inherit some of Ridge’s stability under rotation.
 * But it can suffer with double shrinkage.
* Beyond the above commonly used regression, you can also look at other models like <b>Bayesian, Ecological and Robust regression</b>.
* Regression regularization methods(Lasso, Ridge and ElasticNet) works well in case of <b>high dimensionality</b> and <b>multicollinearity among the variables</b> in the data set.

* Evaluation Metrics for Regression
 * R-square, Adjusted r-square, AIC, BIC and error term
 * Mallow’s Cp criterion, essentially checks for possible bias in your model, by comparing the model with all possible submodels
 * Cross-validation is the best way to evaluate models used for prediction. Here you divide your data set into two group (train and validate). A simple mean squared difference between the observed and predicted values give you a measure for the prediction accuracy.


-- Enhance Predictive Power

1. Segmentation, test whether and which segmentation methods will improve predictive power: http://www.analyticsvidhya.com/blog/2016/02/guide-build-predictive-models-segmentation/?utm_content=bufferfe535&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer

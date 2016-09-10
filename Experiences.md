This file contains experiences, suggestions when doing data analysis


-- Random Forests

1. Random Forest is a powerful algorithm which holistically takes care of missing values, 
outliers and other non-linearities in the data set. It’s simply a collection of classification trees, 
hence the name ‘forest’. Random Forest is a versatile machine learning method capable of performing both regression and classification tasks. It also undertakes dimensional reduction methods, treats missing values, outlier values and other essential steps of data exploration, and does a fairly good job.
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
11. One of benefits of Random forest which excites me most is, the power of handle large data set with higher dimensionality. It can handle thousands of input variables and identify most significant variables so it is considered as one of the <b>dimensionality reduction methods</b>. Further, the model <b>outputs Importance of variable</b>, which can be a very handy feature.
12. It has an effective method for estimating missing data and maintains accuracy when a large proportion of the data are missing.
13. It has methods for balancing errors in data sets where classes are imbalanced.
14. The capabilities of the above can be extended to unlabeled data, leading to unsupervised clustering, data views and outlier detection.
15. It doesn’t predict beyond the range in the training data, and that they may over-fit data sets that are particularly noisy.
16. Random Forest can feel like a black box approach for statistical modelers – you have very little control on what the model does. You can at best – try different parameters and random seeds.
17. Commonly used boosting methods - GBM and Xgboost. Advantages of Xgboost over GBM:
 * Regularization:
Standard GBM implementation has no regularization like XGBoost, therefore it also helps to reduce overfitting.
In fact, XGBoost is also known as ‘regularized boosting‘ technique.
 * Parallel Processing:
XGBoost implements parallel processing and is blazingly faster as compared to GBM.
But hang on, we know that boosting is sequential process so how can it be parallelized? We know that each tree can be built only after the previous one, so what stops us from making a tree using all cores? I hope you get where I’m coming from. (http://zhanpengfang.github.io/418home.html)
XGBoost also supports implementation on Hadoop.
 * High Flexibility
XGBoost allow users to define custom optimization objectives and evaluation criteria.
This adds a whole new dimension to the model and there is no limit to what we can do.
Handling Missing Values
XGBoost has an in-built routine to handle missing values.
User is required to supply a different value than other observations and pass that as a parameter. XGBoost tries different things as it encounters a missing value on each node and learns which path to take for missing values in future.
 * Tree Pruning:
A GBM would stop splitting a node when it encounters a negative loss in the split. Thus it is more of a greedy algorithm.
XGBoost on the other hand make splits upto the max_depth specified and then start pruning the tree backwards and remove splits beyond which there is no positive gain.
Another advantage is that sometimes a split of negative loss say -2 may be followed by a split of positive loss +10. GBM would stop as it encounters -2. But XGBoost will go deeper and it will see a combined effect of +8 of the split and keep both.
 * Built-in Cross-Validation
XGBoost allows user to run a cross-validation at each iteration of the boosting process and thus it is easy to get the exact optimum number of boosting iterations in a single run.
This is unlike GBM where we have to run a grid-search and only a limited values can be tested.
 * Continue on Existing Model
User can start training an XGBoost model from its last iteration of previous run. This can be of significant advantage in certain specific applications.
GBM implementation of sklearn also has this feature so they are even on this point.


* Tree Based Models: http://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
* XGBoost R Tutorial: http://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/
* XGBoost Python Tutorial: http://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/


-- Decision Tree

* It requires less data cleaning compared to some other modeling techniques. It is not influenced by outliers and missing values to a fair degree.
* It can handle both numerical and categorical variables.
* Decision tree is considered to be a non-parametric method. This means that decision trees have no assumptions about the space distribution and the classifier structure.
* Over fitting is one of the most practical difficulty for decision tree models. This problem gets solved by setting constraints on model parameters and pruning.
* While working with continuous numerical variables, decision tree looses information when it categorizes variables in different categories.
* In case of regression tree, the value obtained by terminal nodes in the training data is the mean response of observation falling in that region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mean value.
* In case of classification tree, the value (class) obtained by terminal node in the training data is the mode of observations falling in that region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mode value.
* It is known as <b>greedy</b> because, the algorithm cares (looks for best variable available) about only the current split, and not about future splits which will lead to a better tree.
* The creation of sub-nodes increases the homogeneity of resultant sub-nodes. In other words, we can say that purity of the node increases with respect to the target variable. Decision tree splits the nodes on all available variables and then selects the split which results in most homogeneous sub-nodes.
* 4 Algorithms used for split:
 * Gini Index - It works with categorical target variable. It performs only Binary splits. Higher the value of Gini higher the homogeneity. CART (Classification and Regression Tree) uses Gini method to create binary splits.
 * Chi Square - It works with categorical target variable. It can perform two or more splits. Higher the value of Chi-Square higher the statistical significance of differences between sub-node and Parent node, choose the variable with the highest Chi Square value for splitting. It generates tree called CHAID (Chi-square Automatic Interaction Detector).
 * Information Gain - We build a conclusion that less impure node requires less information to describe it. And, more impure node requires more information. Information theory is a measure to define this degree of disorganization in a system known as <b>Entropy</b>. If the sample is completely homogeneous, then the entropy is zero and if the sample is an equally divided (50% – 50%), it has entropy of one. chooses the split which has lowest entropy compared to parent node and other splits. Entropy is also used with categorical target variable. <b>Information Gain</b> = 1- Entropy
 * Reduction in Variance - Used for continuous target variables (regression problems). 
* Tree models vs. Linear models
 * If the relationship between dependent & independent variable is well approximated by a linear model, linear regression will outperform tree based model.
 * If there is a high non-linearity & complex relationship between dependent & independent variables, a tree model will outperform a classical regression method.
 * If you need to build a model which is easy to explain to people, a decision tree model will always do better than a linear model. Decision tree models are even simpler to interpret than linear regression!


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
* Regression is a <b>parametric approach</b>. ‘Parametric’ means it makes assumptions about data for the purpose of analysis. Due to its parametric side, regression is restrictive in nature.
* Important assumptions in regression analysis:
 * There should be a linear and additive relationship between dependent (response) variable and independent (predictor) variable(s). A linear relationship suggests that a change in response Y due to one unit change in X¹ is constant, regardless of the value of X¹. An additive relationship suggests that the effect of X¹ on Y is independent of other variables.
 * There should be no correlation between the residual (error) terms. Absence of this phenomenon is known as Autocorrelation.
 * The independent variables should not be correlated. Absence of this phenomenon is known as multicollinearity.
 * The error terms must have constant variance. This phenomenon is known as homoskedasticity. The presence of non-constant variance is referred to heteroskedasticity.
 * The error terms must be normally distributed.
* Validate regresssion assumptions and solve the problem when the assumption has been violated
 * Understand regression plots and validate assumptions: http://www.analyticsvidhya.com/blog/2016/07/deeper-regression-analysis-assumptions-plots-solutions/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
 * R plots to validate assumptions and log reansformation to solve the problem: http://www.analyticsvidhya.com/blog/2016/02/complete-tutorial-learn-data-science-scratch/#five


-- Logistic Regression

 * It is widely used for classification problems
 * Logistic regression doesn’t require linear relationship between dependent and independent variables.  It can handle various types of relationships because it applies a non-linear log transformation to the predicted odds ratio
 * To avoid over fitting and under fitting, we should include all significant variables. A good approach to ensure this practice is to use a <b>step wise method</b> to estimate the logistic regression
 * It requires large sample sizes because maximum likelihood estimates are less powerful at low sample sizes than ordinary least square
 * The independent variables should not be correlated with each other i.e. no multi collinearity.  However, we have the options to include interaction effects of categorical variables in the analysis and in the model.
 * If the values of dependent variable is ordinal, then it is called as Ordinal logistic regression 
 * If dependent variable is multi class then it is known as Multinomial Logistic regression.


-- Other Commonly Used Regression

* <b>Polynomial Regression</b> - A regression equation is a polynomial regression equation if the power of independent variable is more than 1. The aim of this modeling technique is to maximize the prediction power with minimum number of predictor variables.
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
2. Inspiration form decision tree, enhance Logistic Regression performance: http://www.analyticsvidhya.com/blog/2013/10/trick-enhance-power-regression-model-2/


-- Neural Network (NN)

* Knowledge behind NN: http://www.analyticsvidhya.com/blog/2016/03/introduction-deep-learning-fundamentals-neural-networks/


-- Ensembling

* Ensembling General: http://www.analyticsvidhya.com/blog/2015/09/questions-ensemble-modeling/
* Simple way to do ensembling with NN: http://www.analyticsvidhya.com/blog/2015/08/optimal-weights-ensemble-learner-neural-network/
* XGBoosing
 * Extreme Gradient Boosting (xgboost) is similar to gradient boosting framework but more efficient. It has both linear model solver and tree learning algorithms. So, what makes it fast is its capacity to do parallel computation on a single machine. It supports various objective functions, including regression, classification and ranking.
 * XGBoost only works with numeric vectors. A simple method to convert categorical variable into numeric vector is One Hot Encoding.
 * xgboost with R example: http://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/
 * xgboost with Python example: http://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
* GBM Param Tuning
 * Boosting algorithms play a crucial role in dealing with bias variance trade-off.  Unlike bagging algorithms, which only controls for high variance in a model, boosting controls both the aspects (<b>bias & variance</b>), and is considered to be more effective
 * Param max_features - As a thumb-rule, square root of the total number of features works great but we should check upto 30-40% of the total number of features.
 * Param presort - Select whether to presort data for faster splits.
 * Params need to be tuned through cross validation: n_estimators, max_depth, min_samples_split



-- Naive Bayesian
* <b>PROS</b>
 * It is easy and fast to predict class of test data set. It also perform well in multi class prediction
 * When assumption of independence holds, a Naive Bayes classifier performs better compare to other models like logistic regression and you need less training data.
 * It perform well in case of categorical input variables compared to numerical variable(s). For numerical variable, normal distribution is assumed (bell curve, which is a strong assumption).
* <b>CONS</b>
 * If categorical variable has a category (in test data set), which was not observed in training data set, then model will assign a 0 (zero) probability and will be unable to make a prediction. This is often known as <b>“Zero Frequency”</b>. To solve this, we can use the smoothing technique. One of the simplest smoothing techniques is called Laplace estimation.
 * On the other side naive Bayes is also known as a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously.
 * Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent.
* <b>Scikit-Learn Bayesian Models</b>
 * Gaussian: It is used in classification and it assumes that features follow a normal distribution.
 * Multinomial: It is used for discrete counts. For example, let’s say,  we have a text classification problem. Here we can consider bernoulli trials which is one step further and instead of “word occurring in the document”, we have “count how often word occurs in the document”, you can think of it as “number of times outcome number x_i is observed over the n trials”.
 * Bernoulli: The binomial model is useful if your feature vectors are binary (i.e. zeros and ones). One application would be text classification with ‘bag of words’ model where the 1s & 0s are “word occurs in the document” and “word does not occur in the document” respectively.
 * If continuous features do not have normal distribution, we should use transformation or different methods to convert it in normal distribution.
* <b>Improve Naive Bayes Model</b>
 * If test data set has zero frequency issue, apply smoothing techniques “Laplace Correction” to predict the class of test data set.
 * Remove correlated features, as the highly correlated features are voted twice in the model and it can lead to over inflating importance.
 * Naive Bayes classifiers has limited options for parameter tuning like alpha=1 for smoothing, fit_prior=[True|False] to learn class prior probabilities or not and some other options (look at detail here). I would recommend to focus on your  pre-processing of data and the feature selection.
 * You might think to apply some classifier combination technique like ensembling, bagging and boosting but these methods would not help. Actually, “ensembling, boosting, bagging” won’t help since their purpose is to reduce variance. Naive Bayes has no variance to minimize.
* <b> Majorly Used Scenarios</b>
 * <b>Real time Prediction</b>: Naive Bayes is an eager learning classifier and it is sure fast. Thus, it could be used for making predictions in real time.
 * <b>Multi class Prediction</b>: This algorithm is also well known for multi class prediction feature. Here we can predict the probability of multiple classes of target variable.
 * <b>Text classification/ Spam Filtering/ Sentiment Analysis</b>: Naive Bayes classifiers mostly used in text classification (due to better result in multi class problems and independence rule) have higher success rate as compared to other algorithms. As a result, it is widely used in Spam filtering (identify spam e-mail) and Sentiment Analysis (in social media analysis, to identify positive and negative customer sentiments)
 * <b>Recommendation System</b>: Naive Bayes Classifier and Collaborative Filtering together builds a Recommendation System that uses machine learning and data mining techniques to filter unseen information and predict whether a user would like a given resource or not



-- Calibration (adjustment)

* Experiments have shown that maximum margin methods such as SVM, boosted trees etc push the real posterior probability away from 0 and 1 while methods such as Naive Bayes tend to push the probabilities towards 0 and 1. And in cases where predicting the accurate probabilities is more important, this poses a serious problem.
* Boosted trees, Random Forests and SVMs performs best after calibration. 
* 2 methods of calibrating the posterior probabilities – <b>Platt Scaling</b> and <b>Isotonic Regression</b>
* Reliability Plots - be used to visualize calibration. On real problems where the true conditional probabilities are not known, model calibration can be visualized with reliability diagrams (DeGroot & Fienberg, 1982). First, the prediction space is discretized into ten bins. Cases with predicted value between 0 and 0.1 fall in the first bin, between 0.1 and 0.2 in the second bin, etc. For each bin, the mean predicted value is plotted against the true fraction of positive cases. If the model is well calibrated the points will fall near the diagonal line.
* The most important point to be noted here is, besides Logloss, other metrics like accuracy, AUC etc are not influenced to an appreciable extent using the Platt Scaling.
* How <b>Platt Scaling</b> works:
 1. Split the train data set into training set and Cross Validation set
 2. Train the model on the training data set
 3. Score test data set and Cross Validation data set
 4. Run a logistic model on the Cross Validation data set using the actual label variable and the predicted values.
 5. Score the test data set using the model created in step 4 with feature as the output of scoring on test data set in step 3.
* <b>Isotonic Regression</b> is similar to Platt Scaling. It’s a non-parametric regression technique. Non-parametric means that it doesn’t make any assumptions such as of linearity among variables, constant error variance etc. The only difference lies in the function being fit. The function we fit in isotonic regression continuously increases/decreases. 
* Reference: http://www.analyticsvidhya.com/blog/2016/07/platt-scaling-isotonic-regression-minimize-logloss-error/?utm_content=buffer2f3d5&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer



-- Time Series Modeling

* <b>Stationary Series</b>: The mean of the series is a constand; The variance of the series should not a be a function of time (also known as homoscedasticity); The covariance of the i th term and the (i + m) th term should not be a function of time
* Stationary Series is important for time series modeling because, when the stationary criterion are violated, the first requisite becomes to <b>stationarize the time series</b> and then try stochastic models to predict this time series. 
* Dickey Fuller Test of Stationarity: <b> X(t) - X(t-1) = (Rho - 1) X(t - 1) + Er(t) </b>  We have to test if Rho – 1 is significantly different than zero or not. If the null hypothesis gets rejected, we’ll get a stationary time series. which means if there is significant difference, we get a stationary time series
* <b>Null Hypothesis</b>: (in a statistical test) the hypothesis that there is no significant difference between specified populations, any observed difference being due to sampling or experimental error.
* <b>ARMA model</b>: AR stands for auto-regression and MA stands for moving average. Remember, AR or MA are NOT applicable on non-stationary series. In MA model, noise / shock quickly vanishes with time. The AR model has a much lasting effect of the shock. Covariance between x(t) and x(t-n) is zero for MA models, the covariance of x(t) and x(t-n) gradually declines with n becoming larger in the AR model. This difference gets exploited irrespective of having the AR model or MA model



-- About Correlation

* For a pair of variables which are perfectly dependent on each other, can also give you a zero correlation. <b>Correlation quantifies the linear dependence of two variables. It cannot capture non-linear relationship between two variables.</b>
* Correlation is NOT transitive.
* Pearson Coefficient is sensitive to outliers. Even a single outlier can change the direction of the coefficient.
* Causation does not imply correlation, because causation can also lead to a non-linear relationship.
* Correlation vs. Simple Linear Regression
 * The square of Pearson’s correlation coefficient is the same as the one in simple linear regression.
 * Neither simple linear regression nor correlation answer questions of causality directly.
 * The slope in  a linear regression gives the marginal change in output/target variable by changing the independent variable by unit distance. Correlation has no slope.
 * The intercept in a linear regression gives the value of target variable if one of the input/independent variable is set zero. Correlation does not have this information.
 * Linear regression can give you a prediction given all the input variables. Correlation analysis does not predict anything.
* Pearson vs. Spearman
 * Pearson captures how linearly dependent are the two variables whereas Spearman captures the monotonic behavior of the relation between the variables.
 * As a <b>thumb rule</b>, you should only begin with Spearman when you have some initial hypothesis of the relation being non-linear. Otherwise, we generally try Pearson first and if that is low, try Spearman. This way you know whether the variables are linearly related or just have a monotonic behavior.
* Correlation vs. Co-variance
 * Correlation is simply the normalized co-variance with the standard deviation of both the factors. 
 * Co-variance is very difficult to compare as it depends on the units of the two variable. so, use the normalized one - Correlation
 * <b>REFERENCE</b>: http://www.analyticsvidhya.com/blog/2015/06/correlation-common-questions/?utm_content=buffer28126&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer


-- Preprocessing

* <b>Feature Scaling</b>, normaly scaling continuous varables to [0,1] or use (x - x_min)/(x_max - x_min)
* <b>Feature Standarization</b> (z-score normalization), means having zero mean and unit variance, with μ=0 and σ=1, where μ is the mean (average) and σ is the standard deviation from the mean, z = (x - μ)/σ
 * Elements such as l1 ,l2 regularizer in linear models (logistic comes under this category) and RBF kernel in SVM in objective function of learners assumes that all the features are centered around zero and have variance in the same order.
 * Label Encoding in Scikit-Learn, encode categorical variables into [0, n_classes-1]
* <b>One-Hot Encoding</b>, transforms each categorical feature with n possible values into n binary features, with only one active, all new variable has boolean values (0 or 1)
* <b>Skewness</b> is a measure of asymmetry of distribution. Many model building techniques have the assumption that predictor values are distributed normally and have a symmetrical shape. Therefore, resolving skeness is necessary in date preprocessing for many models

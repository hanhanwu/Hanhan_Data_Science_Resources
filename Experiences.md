This file contains experiences, suggestions when doing data analysis

## How to Use Randomness in Machine Learning
  * Before Training
    * If the data is not time series, shuffle the data before training is a good practice. Since different order could bring different results.
  * Randomness in Models. 
    * Set seed, to make sure you can repeat the same results. But also try different seeds, and use the aggregated results.
    * Random initialization, such as k in k-means. Try different initialization and choose the optimal one.
  * Actions Can Take:
    * set seeds, and try differnt seeds
    * try emsemble methods
    * repeatedly evaluation, don't just choose the best performed model
    * check whether there is data drifting before a new evaluation period
    * Check distribution of evaluation erros, normally the closer to Garssian, the better
  * reference: https://machinelearningmastery.com/randomness-in-machine-learning/


## Random Forests
* Random Forest is a collection of classification trees, hence the name ‘forest’. It's a bagging method and randomly choose a subset of features in each tree. It also undertakes dimensional reduction methods, treats missing values, outlier values and other essential steps of data exploration, and does a fairly good job.
* Random forest has a feature of presenting the important variables.
* In R library(randomForest), method = “parRF”. This is parallel implemenattion of random forest. 
* Using Random Forests in R, Python: 
http://www.analyticsvidhya.com/blog/2015/09/random-forest-algorithm-multiple-challenges/
* Tuning Random Forests parameters (Python Scikit-Learn): http://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/
* <b>Be careful that random forests have a tendency to bias towards variables that have more number of distinct values. For example, it favors numeric variables over binary/categorical values.</b>
* One of the benefits of Random forest is the power of handle large data set with higher dimensionality. It can handle thousands of input variables and identify most significant variables so it is considered as one of the <b>dimensionality reduction methods</b>. Further, the model <b>outputs Importance of variable</b>, which can be a very handy feature.
  * But I found highly correlated features can all be put as most important features, sometimes it's better to remove highly correlated features before modeling.
* It has an effective method for estimating missing data and maintain accuracy when a large proportion of the data is missing.
* It has methods for balancing errors in data sets where classes are imbalanced.
* The capabilities of the above can be extended to unlabeled data, leading to unsupervised clustering, data views and outlier detection.
* It doesn’t predict beyond the range in the training data, and that they may overfit data sets that are particularly noisy.
* Random Forest can feel like a black box approach for statistical modelers – you have very little control on what the model does. You can at best – try different parameters and random seeds.
### Tips to Tune Random Forest
* n_estimators = number of trees in the foreset. Higher better performance but slower.
* max_features = max number of features considered for splitting a node. Higher value might improve the performance at each node but also reduces the diversity in each tree and also drop the performance.
* max_depth = max number of levels in each decision tree
* min_samples_split = min number of data points placed in a node before the node is split
* min_samples_leaf = min number of data points allowed in a leaf node. Smaller value tends to capture noise better.
* bootstrap = method for sampling data points (with or without replacement)
* oobs_score, set to True then it will use this valication method, similar to leave one out but faster.

## GBM vs Xgboost. 
### Advantages of Xgboost over GBM
* Regularization:
Standard GBM implementation has no regularization like XGBoost, therefore XGBoost also helps to reduce overfitting. In fact, XGBoost is also known as ‘regularized boosting‘ technique.
* Parallel Processing:
  * XGBoost implements parallel processing and is blazingly faster as compared to GBM.
  * XGBoost also supports implementation on Hadoop.
* High Flexibility
  * XGBoost allows customized evaluation functions.
  * See XGBoost Custom Objective and Evaluation Metric: https://xgboost.readthedocs.io/en/latest/tutorials/custom_metric_obj.html
* Handling Missing Values
  * XGBoost has an in-built routine to handle missing values. <b>User is required to supply a different value than other observations and pass that as a parameter.</b> XGBoost tries different things as it encounters a missing value on each node and learns which path to take for missing values in future.
* Tree Pruning:
  * A GBM would stop splitting a node when it encounters a negative loss in the split. Thus it is more of a greedy algorithm.
  * XGBoost on the other hand make splits upto the max_depth specified and then start pruning the tree backwards and remove splits beyond which there is no positive gain.
  * Another advantage is that sometimes a split of negative loss say -2 may be followed by a split of positive loss +10. GBM would stop as it encounters -2. But XGBoost will go deeper and it will see a combined effect of +8 of the split and keep both.
* Built-in Cross-Validation
  * XGBoost allows user to run a cross-validation at each iteration of the boosting process and thus it is easy to get the exact optimum number of boosting iterations in a single run.
  * This is unlike GBM where we have to run a grid-search and only a limited values can be tested.
* Continue on Existing Model
  * For both GBM and XGBoost, user can start training an XGBoost model from its last iteration of previous run. This can be of significant advantage in certain specific applications.
### Tips for Param Tuning
* Param <b>max_features</b> - As a thumb-rule, <b>square root of the total number of features</b> works great but we should check upto 30-40% of the total number of features.
#### Tune GBM
* GBM Param Tuning
  * Param presort - Select whether to presort data for faster splits.
  * Params need to be tuned through cross validation: n_estimators, max_depth, min_samples_split
#### Tune XGBoost
* We can use GridSearch, RandomSearch or python Hyperopt for the param tuning below.
* Step 1 - Tune Tree based params with fixed learning rate and the number of estimators
  * `max_depth`: Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
  * `min_child_weight`: Minimum sum of instance weight needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. Higher the value, less likely to do further spliting (more conservative).
  * `gamma`: alias "min_split_loss". The min loss reduction when spliting a node. Larger the value, more conservation.
  * `subsample`: Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration.
  * `colsample_bytree`: is the subsample ratio of columns when constructing each tree. Subsampling occurs once for every tree constructed. Similar to random forest.
    * It also has a family of parameters `colsample_bylevel`, `colsample_bynode`. "colsample_by*".
* Step 2 - Regularization
  * `reg_alpha`: L1 regularization term on weights. Increasing this value will make model more conservative.
  * `reg_lambda`: L2 regularization term on weights. Increasing this value will make model more conservative.
* Step 3 - Tune Learning Rate
  * Tune `eta`, learning rate
### Notes in Implementation
* XGBoost only works with numeric vectors. A simple method to convert categorical variable into numeric vector is One Hot Encoding.
  * xgboost with R example: http://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/
    * In R, if you simply convert categorical data into numerical with `as.numeric()`, sometimes can get good results too.
  * xgboost with Python example: http://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
### References
* Tree Based Models: http://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
* XGBoost R Tutorial: http://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/
* XGBoost Python Tutorial: http://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/


## More about Ensembling
### Bagging (Bootstrap Aggregating). 
* With Bootstrap, each row is selected with <b>equal probability with replacement</b>. The main purpose of this is to reduce variance. 
* Random forest is also a type of bagging, and it does further variance reducing by <b>randomly selecting a set of features which are used to decide the best split at each node of the decision tree</b>. Random forest uses a decision tree for each subset, and the final result is calculated by averaging all the decision trees. Therefore, in sklearn, you can find random forest has `criteria` to allow you select the method for best spliting, such as Gini, Entropy, etc.
### Boosting
* Boosting algorithms play a crucial role in dealing with bias variance trade-off.  Unlike bagging algorithms, which only controls for high variance in a model, boosting controls both <b>bias & variance</b>, and is considered to be more effective
* Building multiple models (typically of the same type) sequentially, each of which learns to fix the prediction errors of a prior model in the chain (one model one time in this sequence). Previous observations which got incorrectly predicted will be given higher weights and the next model will try to fix the previous errors. The first algorithm of boosting trains on the entire data, later algorithms add higher weights to those pooly predicted observations in the previous model. <b>Each model could be a weak learner for the entire dataset, but it can be good for part of the dataset. The final model is the weighted mean of all the models</b>. In this way, the whole process boosts the performance. However, this may lead to overfitting. Therefore, parameter tuning and cross validation are very important to avoid overfitting in boosting.
* Boosting attempts to minimize residual error which reduces margin distribution.
  * AdaBoost - it normally uses decision tree as the base model. It will stop when error function stays the same or n_estimator has been reached.
  * GBM (Gradient Boosting) - regression trees are used as base models.
  * XGBoost (Extreme Gradient Boosting) - almost 10 times faster than other boosting method, it has regularization to deal with overfitting, it also handles missing data itself. XGBoost makes splits up to the max_depth specified and then starts pruning the tree backwards and removes splits beyond which there is no positive gain.
  * LightGBM - it beats other algorithms when the data isextremely large. It's faster than other algorithms. Leaf-wise while other algorithms are level-wise. Leaf-wise tend to cause overfitting on smaller dataset, you can use `max_depth` to try to avoid overfitting
     * `num_leaves` must be smaller than `2^(max_depth)`, otherwise, it may lead to overfitting.
     * The param descriptions are good: 
       * http://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
       * http://lightgbm.readthedocs.io/en/latest/Parameters.html
  * CatBoost - CatBoost can automatically deal with categorical variables
### Stacking. 
* Building multiple models (typically of differing types) and supervisor model that learns how to best combine the predictions of the primary models. There are multiple layers in Stacking, the lower layers send their precition results to the above layer as features, <b>multiple model predictions are not highly correlated</b>. The top layer can also be Average/Majority Vote/Weighted Average
* <b>If you have m base models in stacking, that will generate m features for second stage models</b>. Because when you have m base models, each model will make the prediction, each prediction result will become the feature column in the next stage. That's why in the next stage, you will have m features.
* <b>Stacking is using cross-validation</b>. For example it's using k-fold cross validation, in each fold, we are using decision tree and KNN as the base models, they are trained on the training data and make predictions on validation data and testing data.  For each base model, validation predictions in each fold will be appended together (append rows), testing predictions will also be appended together. For multiple base models, each appended prediction is a column of feature in the next stage. For the next stage, <b>validation predictions will the training data while testing predictions will be the testing data</b>.
### Blending
* Similar to Stacking, but while stacking uses cross validation, <b>blending is using hold-out</b>.
* With hold-out method, blending has base models to train on training data, predicting on testing data and also evaluate on validation data. Then the validation predictions from all base models will become next stage training data, while testing predictions will become the next stage testing data.
* Both Blending and Stacking use validation predictions as next training data and testing predictions as next testing data
* Consider the confusion caused by the reference below, in real world practice, I can try the code below first, then for both blending and stacking, add original features into the next stage features too, and evaluate the results.
* Here's the code for Stacking and Blending
  ![stacking vs blending](https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/stacking_blending.png)
### Ensembling Types
 * Averaging scores
 * Majority Vote
 * Weighted Average
### Pros & Cons
 * Emsembling can capture both linear and non-linear relationships
 * Reduces the model interpretability
 * May not be good for real-time applications, since it takes longer time
 * It's an art to select models
* NOTE: if the models have highly correlated prediction results, using them together may not bring better results.
### My Code
* DIY Ensembling in R: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/DIY_ensembling.R
### References
* Ensembling General: http://www.analyticsvidhya.com/blog/2015/09/questions-ensemble-modeling/
* Intro to emsembling: https://www.analyticsvidhya.com/blog/2017/02/introduction-to-ensembling-along-with-implementation-in-r/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* The math behind boosting: https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Comprehensive guide to emsemble: https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * I like the example it uses to explain how GBM works (errors became the target of the next model, and the goal is to minimize the error), since I was not sure how did incorrectly predicted results got higher weight and the next model will try to fix the errors
* R Ensembling examples: http://machinelearningmastery.com/machine-learning-ensembles-with-r/


## Decision Tree
* It requires less data cleaning compared to some other modeling techniques. It is <b>not influenced by outliers and missing values</b> to a fair degree.
* It can handle both numerical and categorical variables.
* Decision tree is considered to be a non-parametric method. This means that decision trees have no assumptions about the space distribution and the classifier structure.
* Over fitting is one of the most practical difficulty for decision tree models. This problem gets solved by setting constraints on model parameters and pruning.
* While working with continuous numerical variables, decision tree looses information when it categorizes variables in different categories.
* In case of regression tree, the value obtained by terminal nodes in the training data is the <b>mean response of observation</b> falling in that region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mean value.
* In case of classification tree, the value (class) obtained by terminal node in the training data is the <b>mode of observations</b> falling in that region. Thus, if an unseen data observation falls in that region, we’ll make its prediction with mode value.
* It is known as <b>greedy</b> because, the algorithm cares (looks for best variable available) about only the current split, and not about future splits which will lead to a better tree.
* The creation of sub-nodes increases the homogeneity of resultant sub-nodes. In other words, we can say that purity of the node increases with respect to the target variable. Decision tree splits the nodes on all available variables and then selects the split which results in most homogeneous sub-nodes.
### 5 Algorithms used for split:
* Gini Index - It works with categorical target variable. It performs only Binary splits. Higher the value of Gini higher the homogeneity. CART (Classification and Regression Tree) uses Gini method to create binary splits.
* Chi Square - It works with categorical target variable. It can perform two or more splits. Higher the value of Chi-Square higher the statistical significance of differences between sub-node and Parent node, choose the variable with the highest Chi Square value for splitting. It generates tree called CHAID (Chi-square Automatic Interaction Detector).
* Information Gain - We build a conclusion that less impure node requires less information to describe it. And, more impure node requires more information. Information theory is a measure to define this degree of disorganization in a system known as <b>Entropy</b>. If the sample is completely homogeneous, then the entropy is zero and if the sample is an equally divided (50% – 50%), it has entropy of one. chooses the split which has lowest entropy compared to parent node and other splits. Entropy is also used with categorical target variable.
* Gain Ratio - Information Gain has bias towards the attributes that have larger amount of different values, since this will lead to higher number of branches and each branch is pure. This could make the algorithm useless. To overcome this problem, Gain Ratio has been used
* Reduction in Variance - Used for continuous target variables (regression problems). 
### Tree models vs. Linear models
* If the relationship between dependent & independent variable is well approximated by a linear model, linear regression will outperform tree based model.
* If there is a high non-linearity & complex relationship between dependent & independent variables, a tree model will outperform a classical regression method.
* If you need to build a model which is easy to explain to people, a decision tree model will always do better than a linear model. Decision tree models are even simpler to interpret than linear regression!


## Regression
### Linear Regression
* The line of best fit
  * Sum of square of residuals
  * Sum of the absolute value of residuals
  * NOTE: Sum of residuals will always be zero
* <b>Cost Function</b> = Sum of square of residuals/2m  (m is the number of features in this calculation)
  * To minimize Cost, here comes <b>Gradient Descent</b>
  * <b>Normal Equation</b>, besides gradient descent, normal equation can also be used. In some cases (such as for small feature sets) using it is more effective than applying gradient descent. http://eli.thegreenplace.net/2014/derivation-of-the-normal-equation-for-linear-regression/
* Feature Selection
  * Instead of manually selecting the variables, we can automate this process by using forward or backward selection.
  * <b>Forward selection</b> starts with most significant predictor in the model and adds variable for each step.
  * <b>Backward elimination</b> starts with all predictors in the model and removes the least significant variable for each step. 
  * Selecting criteria can be set to any statistical measure like R-square, t-stat etc.
* Residual Plot
  * Idealy, the plot should looks random, because this part should represent the errors (which is random). Otherwise it means the model missed some determinstic part and should be improved or try non-linear model.
* Linear Regression takes following assumptions:
  * There exists a linear relationship between response (dependent) and predictor (independent) variables
  * The predictor (independent) variables are not correlated with each other. Presence of collinearity leads to a phenomenon known as multicollinearity.
    * Multicollinearity can increase the variance of the coefficient estimates and make the estimates very sensitive to minor changes in the model. The result is that the coefficient estimates are unstable. Use VIF (Variance Inflation Factor) to check multicollineary, and remove those with high VIF (VIF > 10 in practice).
  * The error terms are uncorrelated. Otherwise, it will lead to <b>autocorrelation</b>.
  * Error terms must have constant variance. Non-constant variance leads to <b>heteroskedasticity/heteroscedasticity</b>.
  * Baseline Predition = sum(y)/N, N is the number of records, y is the dependent variable.
* <b>Linear Regression is very sensitive to Outliers</b>. It can terribly affect the regression line and eventually the forecasted values.
* How to implement linear regression: https://towardsdatascience.com/linear-regression-using-python-b136c91bf0a2

### Polynomial Regression
* To deal with non-linear data
* To overcome overfitting, regularization
  * For regularization, when there is more penalty, the model tends to be less complex, so does the decision boundry, the bias can also increase too.
  * Ridge Regression (L2)
      * Higher the values of alpha, bigger is the penalty and therefore the magnitude of coefficients are reduced. If the penalty is large it means model is less complex, therefore the bias would be high.
      * It shrinks the parameters, therefore it is mostly used to prevent multicollinearity.
      * It reduces the model complexity by coefficient shrinkage.
  * LASSO (Least Absolute Shrinkage Selector Operator, L1) 
      * Lasso selects the only some feature while reduces the coefficients of others to zero. This property is known as feature selection and which is absent in case of ridge. But may lose information.
      * It is generally used when we have more number of features, because <b>it automatically does feature selection</b>.
  * Elastic Net Regression - a hybrid of ridge and lasso regression
      * Elastic regression generally works well when we have a big dataset.
      * A combination of both L1 and L2 regularization
      * l1_ratio =1, lasso; l1_ratio =0, ridge; between 0, 1, a combination of lasso and ridge
* As we increase the size of the training data, the bias could increase while the variance could decrease.

### 7 Commonly Used Regression
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
  * This is a regularization method and uses l1 regularization.
  * If group of predictors are highly correlated, lasso picks only one of them and shrinks the others to zero.
* <b>ElasticNet Regression</b> - ElasticNet is hybrid of Lasso and Ridge Regression techniques. It is trained with L1 and L2 prior as regularizer.
  * Elastic-net is useful when there are multiple features which are correlated. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both. It encourages group effect in case of highly correlated variables
  * A practical advantage of trading-off between Lasso and Ridge is that, it allows Elastic-Net to inherit some of Ridge’s stability under rotation.
  * But it can suffer with double shrinkage.
* Beyond the above commonly used regression, you can also look at other models like <b>Bayesian, Ecological and Robust regression</b>.
* Regression regularization methods(Lasso, Ridge and ElasticNet) works well in case of <b>high dimensionality</b> and <b>multicollinearity among the variables</b> in the data set.

### Performance Evaluation Metrics
* SSE - minimum sum of squared errors (SSE), but it <b>highly sensitive to the number of data points</b>.
* R-Square: <b>How much the change in output variable (y) is explained by the change in input variable(x)</b>. Its value is between 0 and 1, 0 indicates that the model explains NIL variability in the response data around its mean, 1 indicates that the model explains full variability in the response data around its mean. R² has less variation in score compare to SSE. One disadvantage of R-squared is that it can only increase as predictors are added to the regression model. This increase is artificial when predictors are not actually improving the model’s fit.
 * Adjusted R-Square: To deal with the disadvantage in R-Square. Adjusted R-squared will decrease as predictors are added if the increase in model fit does not make up for the loss of degrees of freedom. Likewise, it will increase as predictors are added if the increase in model fit is worthwhile. Adjusted R-squared should always be used with models with more than one predictor variable.
* <b>Besides using metrics above, better to plot residual plot at the same time</b>, to check whether the plot is random. If not, the model needs to be improved or need to change to non-linear model.
  * Spark Python has `Fitted vs Residuals plot` for the validaton of both linear regression and logistic regression. A good linear model will usually have <b>residuals distributed randomly around the residuals=0 line</b> with no distinct outliers and no clear trends. The <b>residuals should also be small</b> for the whole range of fitted values.
  
### References
* A Summarization Artile (a very good one!): https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Linear Regression with basic R, Python code: http://www.analyticsvidhya.com/blog/2015/10/regression-python-beginners/
* <b>R Square and Adjusted R Square</b>: https://discuss.analyticsvidhya.com/t/difference-between-r-square-and-adjusted-r-square/264/3
* http://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/


## Logistic Regression
* It is widely used for classification problems.
* In Linear Regression, the output is the weighted sum of inputs. Logistic Regression is a generalized Linear Regression in the sense that we don’t output the weighted sum of inputs directly, but we pass it through a function that can map any real value between 0 and 1.
  * Similar to having an activation function.
  * If we take the weighted sum of inputs as the output as we do in Linear Regression, the value can be more than 1 but we want a value between 0 and 1. That’s why Linear Regression can’t be used for classification tasks.
* Logistic regression doesn’t require linear relationship between dependent and independent variables. It can handle various types of relationships because it applies a non-linear log transformation to the predicted odds ratio. But Logistic Regression only forms a linear decision surface to make the classified targets linearly seperated.
* To avoid over fitting and under fitting, we should include all significant variables. A good approach to ensure this practice is to use a <b>step wise method</b> to estimate the logistic regression. Stepwise selection is a method that allows moves in either direction, dropping or adding variables at the various steps. For example, backward selection, involves starting off in a backward approach and then potentially adding back variables if they later appear to be significant.
* It uses <b>maximum likelihood</b> to best fit the data. Maximum likelihood is the procedure of finding the value of one or more parameters for a given statistics which makes the known likelihood distribution a maximum.
* It <b>requires large sample sizes</b> because maximum likelihood estimates are less powerful at low sample sizes than ordinary least square.
* The independent variables (features) should not be correlated with each other i.e. no multi-collinearity. However, we have the options to include interaction effects of categorical variables in the analysis and in the model.
* If the values of <b>dependent variable is ordinal</b>, then it is called as <b>Ordinal logistic regression</b>.
* If dependent variable is multi-class then it is known as <b>Multinomial Logistic regression</b>.
* To evaluate Logistic Regression, we can use ROC curve, and <b>we can adjust the threshold of ROC curve basde on how much we value True Positive Rate or False Positive Rate</b>.
* AIC: explains the degree to which your input variables explain the variation of your output / predicted variable, similar to R-Squared/Adjusted R-Squared in linear regression.
* Linear Regression errors values has to be normally distributed but in case of Logistic Regression it is not the case.
* To Deal with multi-class problem
  * You can use multinomial logistic regression
  * You can also use one_vs_all method: https://www.coursera.org/learn/machine-learning/lecture/68Pol/multiclass-classification-one-vs-all
    * In this method, if you have k classes, then use k logistic regression models, each model, lable 1 of the classeds as positive and other classes as negative, and generate the probability of positive class in each model.
    * Finally you get the probability for each class.
* How to implement Logistic Regression
  * https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
  * You will learn how to implement: sigmoid function, computing weighted sums, cost function and how to minimize cost function with gradient.


## Enhance Predictive Power
* Ensembling methods
* Segmentation
  * Divide the data into differnt groups, each group train a model and see whether the overall performance would improve.
    * You can use domain knowledge or machine learning methods to divide the data into groups.
  * Segmentation example1: http://www.analyticsvidhya.com/blog/2016/02/guide-build-predictive-models-segmentation/?utm_content=bufferfe535&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
  * Example 2: https://www.analyticsvidhya.com/blog/2013/10/trick-enhance-power-regression-model-2/
* Clustering labels as features
  * Cluster the data, use cluster id as a new feature for classification

## Neural Network (NN)

* Simple way to find optimal weights in NN: http://www.analyticsvidhya.com/blog/2015/08/optimal-weights-ensemble-learner-neural-network/
  * Just use 1 hiden layer.


## Naive Bayesian
* <b>PROS</b>
  * It is easy and fast to predict class of test data set. It also performs well in multi class prediction.
  * When assumption of independence holds, a Naive Bayes classifier performs better compare to other models like logistic regression and you need less training data.
  * It performs well in case of categorical input variables compared to numerical variable(s). For numerical variable, normal distribution is assumed (bell curve, which is a strong assumption).
* <b>CONS</b>
  * If categorical variable has a category (in test data set), which was not observed in training data set, then model will assign a 0 (zero) probability and will be unable to make a prediction. This is often known as <b>“Zero Frequency”</b>. To solve this, we can use the smoothing technique. One of the simplest smoothing techniques is called <b>Laplace Smoothing</b>.
    * Detailed formulas for laplace smoothing: https://towardsdatascience.com/introduction-to-na%C3%AFve-bayes-classifier-fa59e3e24aaf
      * The main idea is to add a small constant in divisor, dividen of prior probability and conditional probability.
  * On the other side Naive Bayes is also known as a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously.
  * Another limitation of Naive Bayes is the assumption of independent predictors. In real life, it is almost impossible that we get a set of predictors which are completely independent.
* <b>Scikit-Learn Bayesian Models</b>
  * <b>Gaussian</b>: It is used in classification and it assumes that features follow a normal distribution.
  * <b>Multinomial</b>: It is used for <b>discrete counts</b>. It deals with “count how often word occurs in the document”, you can think of it as <b>“number of times outcome number x_i is observed over the n trials”</b>.
  * <b>Bernoulli</b>: The binomial model is useful if your feature vectors are binary (i.e. zeros and ones). One application would be text classification with ‘bag of words’ model where the 1s & 0s are “word occurs in the document” and “word does not occur in the document” respectively.
  * If continuous features do not have normal distribution, we should use transformation or different methods to convert it in normal distribution.
* <b>Improve Naive Bayes Model</b>
  * If test data set has zero frequency issue, apply smoothing techniques “Laplace Smoothing” to predict the class of test data set.
  * Remove correlated features, as the highly correlated features are voted twice in the model and it can lead to over inflating importance.
  * Naive Bayes classifiers has limited options for parameter tuning like alpha=1 for smoothing, fit_prior=[True|False] to learn class prior probabilities or not and some other options. I would recommend to focus on your pre-processing of data and the feature selection.
  * You might think to apply some classifier combination technique like ensembling <b>but these methods would not help</b>. Actually, ensembling won’t help <b>since their purpose is to reduce variance. Naive Bayes has no variance to minimize</b>.
* <b> Majorly Used Scenarios</b>
  * <b>Real time Prediction</b>: Naive Bayes is an eager learning classifier and it is very fast. Thus, it could be used for making predictions in real time.
  * <b>Multi class Prediction</b>: This algorithm is also well known for multi class prediction feature. Here we can predict the probability of multiple classes of target variable.
  * <b>Text classification/ Spam Filtering/ Sentiment Analysis</b>: Naive Bayes classifiers mostly used in text classification (due to better result in multi class problems and independence rule) have higher success rate as compared to other algorithms. As a result, <b>it is widely used in Spam filtering (identify spam e-mail) and Sentiment Analysis (in social media analysis, to identify positive and negative customer sentiments)</b>.
  * <b>Recommendation System</b>: Naive Bayes Classifier and Collaborative Filtering together builds a Recommendation System that uses machine learning and data mining techniques to filter unseen information and predict whether a user would like a given resource or not.


## SVM
* Objects that are close to the margins are Supporting Vectors, the margins will only be influenced by supporting vectors, not other objects. The goal of SVM is to find the maximum margin that can seperate 2 classes.
* How does SVM find the right hyper-plane
  * First of all, it chooses the hyper-plane which seperate 2 classes with lowest mis-classification, this is prior to choosing the one with the highest margin.
  * Choose the one with the highest margin, which maximizes the distance between the hyper-plane and its cloest data points
  * SVM has a feature to <b>ignore outliers</b> when chosing hyper-plane
  * SVM also works for non-linear seperation problem. With <b>Kernel Trick</b> technique, SVM is able to transform </b>non-linear problem into </b>linear problem</b>, by converting lower dimentional input space into higher dimensional space, because when it's in a higher dimension space, the 2 classes might be linearly seperable.
* SVM params in Python Scikit-Learn, “kernel”, “gamma” and “C”.
  * <b>kernel</b>: We have values such as “linear”, “rbf”,”poly” and others (default value is “rbf”, radial based function).  Here “rbf” and “poly” are useful for non-linear hyper-plane.
  * <b>gamma</b>: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. Higher the value of gamma, will try to exact fit the as per training data set i.e. generalization error and cause over-fitting problem.
  * <b>C</b>: Penalty parameter C of the error term. It also controls the trade off between smooth decision boundary and classifying the training points correctly. Similar to regularization.
* <b>Pros</b>
  * It works really well with clear margin of separation
  * It also works well on small dataset
  * It is effective in high dimensional spaces.
  * It is effective in cases where number of dimensions is greater than the number of samples.
  * It uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* <b>Cons</b>
  * It doesn’t perform well, when we have large data set because the required training time is higher
  * It also doesn’t perform very well, when the data set has more noise i.e. target classes are overlapping
  * SVM doesn’t directly provide probability estimates, these are calculated using an expensive five-fold cross-validation. It is related SVC method of Python scikit-learn library.
* Reference: https://www.analyticsvidhya.com/blog/2015/10/understaing-support-vector-machine-example-code/?utm_content=buffer02b8d&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer


## Evaluation Metrics in Sampling Experiments
* <b>Cohen's kappa coefficient</b> is a statistic which measures inter-rater agreement for categorical items. Cohen's kappa measures the agreement between two raters who each classify N items into C mutually exclusive categories. https://en.wikipedia.org/wiki/Cohen's_kappa
* <b>Fleiss's kappa</b> assesses the reliability of agreement between a fixed number of raters when assigning categorical ratings to a number of items or classifying items. This contrasts with other kappas such as Cohen's kappa, which only work when assessing the agreement between not more than two raters or the interrater reliability for one appraiser versus themself. The measure calculates the degree of agreement in classification over that which would be expected by chance. Fleiss' kappa can be used <b>only with binary or nominal-scale ratings</b>. No version is available for ordered-categorical ratings. https://en.wikipedia.org/wiki/Fleiss'_kappa
* <b>Krippendorff's alpha</b> measures the agreement achieved when coding a set of units of analysis in terms of the values of a variable. Krippendorff’s alpha is applicable to any number of coders, each assigning one value to one unit of analysis, to incomplete (missing) data, to any number of values available for coding a variable, to binary, nominal, ordinal, interval, ratio, polar, and circular metrics (Levels of Measurement), and it adjusts itself to small sample sizes of the reliability data. The virtue of a single coefficient with these variations is that computed reliabilities are comparable across any numbers of coders, values, different metrics, and unequal sample sizes. https://en.wikipedia.org/wiki/Krippendorff's_alpha


## About Correlation
* For a pair of variables which are perfectly dependent on each other, can also give you a zero correlation. <b>Correlation quantifies the linear dependence of two variables. It cannot capture non-linear relationship between two variables.</b>
* Correlation is NOT transitive!
  * I like the rescription here: https://stats.stackexchange.com/questions/181376/is-correlation-transitive
* Pearson Coefficient is sensitive to outliers. Even a single outlier can change the direction of the coefficient.
* Causation does not imply correlation, because causation can also lead to a non-linear relationship.
* Correlation vs. Simple Linear Regression
 * The square of Pearson’s correlation coefficient is the same as the one in simple linear regression.
 * Neither simple linear regression nor correlation answer questions of causality directly.
 * The slope in a linear regression gives the marginal change in output/target variable by changing the independent variable by unit distance. Correlation has no slope.
 * The intercept in a linear regression gives the value of target variable if one of the input/independent variable is set to zero. Correlation does not have this information.
 * Linear regression can give you a prediction given all the input variables. Correlation analysis does not predict anything.
* Pearson vs. Spearman
 * Pearson captures how linearly dependent are the two variables whereas Spearman captures the monotonic behavior of the relation between the variables. Monotonic can be nonlinear.
 * As a <b>thumb rule</b>, you should only begin with Spearman when you have some initial hypothesis of the relation being non-linear. Otherwise, we generally try Pearson first and if that is low, try Spearman. This way you know whether the variables are linearly related or just have a monotonic behavior.


## Clustering
* When the data reocrds are small, it's better to use Capping & Flooring, Bucketing to deal with outliers, instead of removing them.
* Kmeans
  * It is better to run K-Means clustering multiple times before using the generated clusters, <b>when the K-Means algorithm has reached the local or global minima, it will not alter the assignment of data points to clusters for two successive iterations.</b>
  * K-means fails when there are outliers, density spread of the data points and data points have non-convex shapes.
  * K-means is also sensitive to initialization, bad initialization will lead to poor convergence speed as well as overall clustering performance.
  * K-Means clustering algorithm and EM clustering algorithm has the drawback of converging at local minima. Agglomerative clustering algorithm (hierarchical clustering) and Diverse clustering algorithm do not have this drawback.
  * 2 K-Means initialization methods:
    * Forgy Method - Choose k data points as initial centers
    * Random Partition - random assign a cluster to each data point

* About EM Clustering: http://docs.rapidminer.com/studio/operators/modeling/segmentation/expectation_maximization_clustering.html
  * It is also known as Gausian Mixture Model.
  * EM Clsuering is similar to K-Means, but it calculates the probability of cluster membership and tries to maximum the overall likelihood of the data. Unlike K-Means, <b>EM Clustering applies to both numerical and categorical data</b>.
  * When using EM clustering, the assumption is all the data points follow two multinomial distribution.
  * K-Means is a special case of EM algorithm in which only the centroids of the cluster distributions are calculated at each iteration.
  * <b>Both K-Means and EM Clustering make strong assumption of dataset.</b>
  * <b>Soft assignment</b> in Clustering means return cluster membership with probabilities, instead of indiating only 1 cluster. Gausian Mixture Model and Fuzzy K-Means are soft assignment
  
* Hierarchical Clustering
  * A dendrogram (hierarchical relationship) is not possible for K-Means clustering analysis.
  * When you are checking the best number of clusters from a dendrogram generated by hierarchical clustering, the method is to draw 2 horizontial lines, and the number of vertical lines between the 2 horizontal lines can transverse the maximum distance vertically without intersecting a cluster.
  * single link, complete link and average link can be used for finding dissimilarity between two clusters in hierarchical clustering
    * Single-Link vs Complete-Link from Stanford NLP: https://nlp.stanford.edu/IR-book/html/htmledition/single-link-and-complete-link-clustering-1.html
    * Single-Link's merge criterion is local, while Complete-Link is non-local
    
* <b>heteroscedasticity</b> - variance in features, this will not influence clustering analysis.
* <b>multicollinearity</b> - features are correlated, this will create negative influence on clustering analysis, since correlated features will carry extra weight on the distance calculation.

* Density Based Clustering - DBSCAN
  * For data points to be in a cluster, they must be in a distance threshold to a core point.
  * DBSCAN can form a cluster of any arbitrary shape and does not have strong assumptions for the distribution of data points in the dataspace.
  * DBSCAN has time complexity of order O(NlogN) if spatial index is used, otherwise it will be O(N^2).
  * It does not require prior knowledge of the number of desired clusters.
  * Robust to outliers.
  
* <b>To sum up necessary data preprocessing methods before clustering</b>
  * check and deal with outliers
  * check and deal with missing data
  * deal with 0 variance features, pay attention to near 0 variance features too
  * check and deal with feature correlation
  * data normalization (scaling) - bring all the features to the same scale
  
* <b>To sum up the process of doing clustering analysis</b>
  * Round 1 - KMeans
    * data preprocessing
    * several iterations of clustering till assigned clusters no longer change
    * elbow method to find optimal number of clusters
    * if there is no clear elbow, visualization SSE and silhouette coefficient together, choose the cluster number that has higher silhouette coefficient and lower SSE
    * silhouette coefficient to compare cluster similarity
  * Round 2 - Gaussian Mixture Model/EM Clustering
    * data preprocessing
    * It does soft assignment
    * Compare membership probabilities of clusters
  * Round 3 - DBSCAN
    * data preprocessing, althoug it is robust to outliers
  * Round 4 - Clustering Ensembling
    

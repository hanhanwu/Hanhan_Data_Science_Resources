# Hanhan_Data_Science_Resources
helpful resources for (big) data science


* Data Preprocessing

 * Data Exploration: http://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/
 * Data Exploration PDF: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/data%20exploration.pdf
 * Faster Data Manipulation with 7 R Packages: http://www.analyticsvidhya.com/blog/2015/12/faster-data-manipulation-7-packages/
 * Dimension Reduction Methods: http://www.analyticsvidhya.com/blog/2015/07/dimension-reduction-methods/
 * 7 methods to reduce dimensionality: https://www.knime.org/files/knime_seventechniquesdatadimreduction.pdf
 * Important Predictive Model Evaluation Metrics: http://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/
 * using PCA for dimension reduction [R and Python]: http://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/?utm_content=buffer40497&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 * Why using One Hot encoding to convert categorical data into numerical data and only choose the top N columns after using PCA is right: http://stats.stackexchange.com/questions/209711/why-convert-categorical-data-into-numerical-using-one-hot-encoding
 * Using PLS for dimension reduction and prediction: http://www.r-bloggers.com/partial-least-squares-regression-in-r/
 * Instead of using PCA, using Random Forests to add chosen features: http://myabakhova.blogspot.ca/2016/04/improving-performance-of-random-forests.html
 * Data Sampling methods to deal with inbalanced dataset for classification: http://www.analyticsvidhya.com/blog/2016/03/practical-guide-deal-imbalanced-classification-problems/?utm_content=buffer929f7&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 

*******************************************************

* R

 * R Basics: http://www.analyticsvidhya.com/blog/2016/02/complete-tutorial-learn-data-science-scratch/
 * Code for R Basics: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/R_Basics.R
 * Data Set for R Basics: http://datahack.analyticsvidhya.com/contest/practice-problem-bigmart-sales-prediction
 * Interesting R Librarise Graph: http://www.analyticsvidhya.com/blog/2015/08/list-r-packages-data-analysis/
 * R Visualization Basics: http://www.analyticsvidhya.com/blog/2015/07/guide-data-visualization-r/
 * Data Visualization Cheatsheet (ggplot2): https://www.rstudio.com/wp-content/uploads/2015/03/ggplot2-cheatsheet.pdf
 * data.table, much fater than data.frame: http://www.analyticsvidhya.com/blog/2016/05/data-table-data-frame-work-large-data-sets/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
 * Data Modeling with H2O, with R data.table: http://www.analyticsvidhya.com/blog/2016/05/h2o-data-table-build-models-large-data-sets/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
 * H2O.ai: http://www.h2o.ai/
 

********************************************************

CLOUD PLATFORM MACHINE LEARNING

* Azure Machine Learning
 
 * AzureML Studio Overview (2015 Version): https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/ml_studio_overview_v1.1.pdf
 * Choose Azure ML Algorithms Cheatsheet:
 https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/microsoft-machine-learning-algorithm-cheat-sheet-v6.pdf
 * How to choose Azure ML Algorithms:
 https://azure.microsoft.com/en-us/documentation/articles/machine-learning-algorithm-choice/
 * How to choose parameters for AzureML Studio Algorithms: https://azure.microsoft.com/en-us/documentation/articles/machine-learning-algorithm-parameters-optimize/
 * AzureML Modules: https://msdn.microsoft.com/en-us/library/azure/dn905870.aspx
 * Compare Azure ML Algorithms Cheatsheet with Python Scikit-Learn ML Map: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/scikit_learn_ML_map.png
 * Want to know more Python Scikit-Learn: http://scikit-learn.org/stable/index.html
 * Azure ML Documentation: https://azure.microsoft.com/en-us/documentation/services/machine-learning/
 * Azure ML Tutorials: http://gallery.cortanaintelligence.com/tutorials
 * Azure ML Experiments Gallary: http://gallery.cortanaintelligence.com/browse?orderby=freshness%20desc
 * Background Algorithms for Azure ML Algorithms: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/AzureML_Algorithms_Notes.md
 * AzureML Studio R Script Tutorial with Time Series Forecasting: https://azure.microsoft.com/en-us/documentation/articles/machine-learning-r-quickstart/
 * Modules for basic data preprocessing: https://azure.microsoft.com/en-us/documentation/videos/preprocessing-data-in-azure-ml-studio/


* Spark

 * Spark Advanced Analysis: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/Mastering-Advanced-Analytics-With-Apache-Spark.pdf
 * Spark 2.0 live presentation: https://www.brighttalk.com/webcast/12891/202021?utm_campaign=communication_reminder_starting_now_registrants&utm_medium=email&utm_source=brighttalk-transact&utm_content=button

********************************************************

* Statistical Methods

 * Hypothesis Tests and relative statistics knowledge: http://www.analyticsvidhya.com/blog/2015/09/hypothesis-testing-explained/
 * z-score and p-value table: http://www.stat.ufl.edu/~athienit/Tables/Ztable.pdf


********************************************************

* Terminology Wiki

 * Multicollinearity: https://en.wikipedia.org/wiki/Multicollinearity
 * Autocorrelation: https://en.wikipedia.org/wiki/Autocorrelation#Regression_analysis
 * Heteroscedasticity: https://en.wikipedia.org/wiki/Heteroscedasticity
 * Bias: how much on an average are the predicted values different from the actual value
 * Variance: how different will the predictions of the model be at the same point if different samples are taken from the same population


********************************************************

* Data Analysis Tricks and Tips

-- ENSEMBLE

 * Ensemble modeling offers one of the most convincing way to build highly accurate predictive models. The availability of bagging and boosting algorithms further embellishes this method to produce awesome accuracy level.
 * Basic Ensemble Modeling: http://www.analyticsvidhya.com/blog/2015/08/introduction-ensemble-learning/
 * Ensemble Modeling: http://www.analyticsvidhya.com/blog/2015/09/questions-ensemble-modeling/
 * In Ensemble Modeling, we can combine multiple models of same ML algorithms, but combining multiple predictions generated by different algorithms would normally give you better predictions. It is due to the diversification or independent nature as compared to each other.
 * AdaBoost and Gradient Boost: http://www.analyticsvidhya.com/blog/2015/05/boosting-algorithms-simplified/
 * Boosting may lead to overfittig if the sample data is too small.
 * All the Ensemble methods (Bagging, Boosting, Stacking) may lead to overfitting.
 * Finding Optimal Weights of Ensemble Learner using Neural Network: http://www.analyticsvidhya.com/blog/2015/08/optimal-weights-ensemble-learner-neural-network/
 * Bagging Sample in R: http://www.analyticsvidhya.com/blog/2015/09/selection-techniques-ensemble-modelling/
 * Differences between bagging, boosting and stacking
 

-- DEAL WITH IMBALANCED DATASET

 * Inbalanced dataset  in classification: http://www.analyticsvidhya.com/blog/2016/03/practical-guide-deal-imbalanced-classification-problems/?utm_content=buffer929f7&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer


********************************************************

* Machine Learning Experiences

https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/Experiences.md


********************************************************

* Good to Read

 * Causality and Correlation: http://www.analyticsvidhya.com/blog/2015/06/establish-causality-events/
 * Why NoSQL: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/IBM%20Q1%20Cloud%20Data%20Services%20Asset%232%20-%20Why%20NoSQL%20ov38975.pdf
 * Data Science Life Cycle: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/IBM%20Q1%20Technical%20Marketing%20ASSET2%20-%20Data%20Science%20Methodology-Best%20Practices%20for%20Successful%20Implementations%20ov37176.pdf
 * Machine Learning Algorithms Overview with R/Pyhton code: http://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/

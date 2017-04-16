# Hanhan_Data_Science_Resources
helpful resources for (big) data science


DATA PREPROCESSING

 * Google OpenRefine for data transformation, matrix pivorting when there are many inconsistency (It has its own fancy, but if you can use R/Python, use them first): [tutorials for beginners][3], [many more tutorials][4], [regex cheatsheet][5], [OpenRefine Language][6]
 * Trifacta for data refinement for small dataset <b>non-private</b> data, it allows you to do data wrangling with interactive user interface, with its Wrangle language, you will have more flexibility to do data preprocessing. Its `unpivot` method is good because tools like Tableau only compiles a certain type of data format, therefore some data wrangling is necessary. (The interactive user interface of this tool is really great, but if you can use R/Python, use them first) [online tutorials][1], [Trifacta Wrangle Language][2]
 * Data Exploration: http://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/
 * Data Exploration PDF: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/data%20exploration.pdf
 * Faster Data Manipulation with 7 R Packages: http://www.analyticsvidhya.com/blog/2015/12/faster-data-manipulation-7-packages/
 * Dimension Reduction Methods: http://www.analyticsvidhya.com/blog/2015/07/dimension-reduction-methods/
 * 7 methods to reduce dimensionality: https://www.knime.org/files/knime_seventechniquesdatadimreduction.pdf
 * 5 R packages to deal with missing values:  http://www.analyticsvidhya.com/blog/2016/03/tutorial-powerful-packages-imputing-missing-values/?utm_content=buffer916b5&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 * Important Predictive Model Evaluation Metrics: http://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/
 * using PCA for dimension reduction [R and Python]: http://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/?utm_content=buffer40497&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 * Why using One Hot encoding to convert categorical data into numerical data and only choose the top N columns after using PCA is right: http://stats.stackexchange.com/questions/209711/why-convert-categorical-data-into-numerical-using-one-hot-encoding
 * Using PLS for dimension reduction and prediction: http://www.r-bloggers.com/partial-least-squares-regression-in-r/
 * Instead of using PCA, using Random Forests to add chosen features: http://myabakhova.blogspot.ca/2016/04/improving-performance-of-random-forests.html
 * Easy simple way to do feature selection with Boruta: http://www.analyticsvidhya.com/blog/2016/03/select-important-variables-boruta-package/?utm_content=bufferec6a6&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 * Data Sampling methods to deal with inbalanced dataset for classification: http://www.analyticsvidhya.com/blog/2016/03/practical-guide-deal-imbalanced-classification-problems/?utm_content=buffer929f7&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 * Deal with continuous variables: http://www.analyticsvidhya.com/blog/2015/11/8-ways-deal-continuous-variables-predictive-modeling/?utm_content=buffer346f3&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 * Deal with categorical variables (combine levels, convert to numerical data): https://www.analyticsvidhya.com/blog/2015/11/easy-methods-deal-categorical-variables-predictive-modeling/
 * Deal with imbalanced data in classification: https://www.analyticsvidhya.com/blog/2016/09/this-machine-learning-project-on-imbalanced-data-can-add-value-to-your-resume/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
 * Pandas Basics: http://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/?utm_content=bufferfa8d9&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 * Common useful operations in R data.frame and Python Pandas DataFrame (add, drop, removing duplicates, modify, rename): http://www.analyticsvidhya.com/blog/2016/06/9-challenges-data-merging-subsetting-r-python-beginner/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
 * <b>Calibration</b> - Minimize Logloss: http://www.analyticsvidhya.com/blog/2016/07/platt-scaling-isotonic-regression-minimize-logloss-error/?utm_content=buffer2f3d5&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 * My R code for minimize logloss: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/minimize_logloss.R
 * <b>Importance of Calibration</b> - In many applications it is important to predict well brated probabilities; good accuracy or area under the ROC curve are not sufficient.
 * A paper about Calibration: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/Predicting%20good%20probabilities%20with%20supervised%20learning.pdf
 * Validate Regression Assumptions: http://www.analyticsvidhya.com/blog/2016/07/deeper-regression-analysis-assumptions-plots-solutions/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
 * Plots to validate Regression assumptions and log transformation to deal with assumption violation: http://www.analyticsvidhya.com/blog/2016/02/complete-tutorial-learn-data-science-scratch/#five
 * Python Scikit-Learn preprocessing methods: http://www.analyticsvidhya.com/blog/2016/07/practical-guide-data-preprocessing-python-scikit-learn/?utm_content=buffera1e2c&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 
 
[1]:https://www.trifacta.com/support/
[2]:https://docs.trifacta.com/display/PE/Wrangle+Language
[3]:https://github.com/OpenRefine/OpenRefine/wiki/Screencasts
[4]:https://github.com/OpenRefine/OpenRefine/wiki/External-Resources
[5]:https://www.cheatography.com/davechild/cheat-sheets/regular-expressions/pdf/
[6]:https://github.com/OpenRefine/OpenRefine/wiki/General-Refine-Expression-Language
 
 
*******************************************************

FEATURE ENGINEERING

* Feature Selection: https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Why Feature Selection:
  * It enables the machine learning algorithm to train faster.
  * It reduces the complexity of a model and makes it easier to interpret.
  * It improves the accuracy of a model if the right subset is chosen.
  * It reduces overfitting.
* <b>Filter Methods</b>, the selection of features is independent of any machine learning algorithms. Features are selected on the basis of their scores in various statistical tests for their <b>correlation with the dependent variable</b>. Example - Pearson’s Correlation, LDA, ANOVA, Chi-Square.
* <b>Wrapper Methods</b>, try to use a subset of features and train a model using them. Based on the inferences that we draw from the previous model, we decide to add or remove features from your subset. These methods are usually computationally very expensive. Example - Forward Selection, Backward Elimination, Recursive Feature elimination.
* <b>Embedded Methods</b>, implemented by algorithms that have their own built-in feature selection methods. Example - LASSO and RIDGE regression. <b>Lasso regression</b> performs L1 regularization which adds penalty equivalent to absolute value of the magnitude of coefficients. <b>Ridge regression</b> performs L2 regularization which adds penalty equivalent to square of the magnitude of coefficients. Other examples of embedded methods are Regularized trees, Memetic algorithm, Random multinomial logit.
* <b> Differences between Filter Methods and Wrapper Methods</b>
  * Filter methods measure the relevance of features by their correlation with dependent variable while wrapper methods measure the usefulness of a subset of feature by actually training a model on it.
  * Filter methods are much faster compared to wrapper methods as they do not involve training the models. On the other hand, wrapper methods are computationally very expensive as well.
  * Filter methods use statistical methods for evaluation of a subset of features while wrapper methods use cross validation.
  * Filter methods might fail to find the best subset of features in many occasions but wrapper methods can always provide the best subset of features.
  * Using the subset of features from the wrapper methods make the model more prone to overfitting as compared to using subset of features from the filter methods.
* <b>My strategy to use feature selection</b>
  * Use filter methods in data preprocessing step, before training. Choose the top features for model training.
  * Use wrapper methods during the model training step.


*******************************************************

DATA MINING BIBLE

* Data Mining book (writtern by the founder of data mining): https://github.com/hanhanwu/readings/blob/master/Data%20Mining.pdf
  * Reading Notes: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/DataMining_Reading_Notes.md


*******************************************************

R

* R Basics: http://www.analyticsvidhya.com/blog/2016/02/complete-tutorial-learn-data-science-scratch/
* Code for R Basics: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/R_Basics.R
* ALL IN ONE - R MLR (a package contains all commonly used algorithms and data preprocessing methods): https://www.analyticsvidhya.com/blog/2016/08/practicing-machine-learning-techniques-in-r-with-mlr-package/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Data Set for R Basics: http://datahack.analyticsvidhya.com/contest/practice-problem-bigmart-sales-prediction
* Interesting R Librarise Graph: http://www.analyticsvidhya.com/blog/2015/08/list-r-packages-data-analysis/
* 7 commonly used R data summary methods: http://www.analyticsvidhya.com/blog/2015/12/7-important-ways-summarise-data/
  * correct R code is here: https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/R_summarize_methods.R

* R Visualization Basics: http://www.analyticsvidhya.com/blog/2015/07/guide-data-visualization-r/
* Data Visualization Cheatsheet (ggplot2): https://www.rstudio.com/wp-content/uploads/2015/03/ggplot2-cheatsheet.pdf
* data.table, much fater than data.frame: http://www.analyticsvidhya.com/blog/2016/05/data-table-data-frame-work-large-data-sets/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Data Modeling with H2O, with R data.table: http://www.analyticsvidhya.com/blog/2016/05/h2o-data-table-build-models-large-data-sets/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* H2O.ai: http://www.h2o.ai/
* Basic methods to deal with continuous variables: http://www.analyticsvidhya.com/blog/2015/11/8-ways-deal-continuous-variables-predictive-modeling/?utm_content=buffer346f3&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* Connect to Oracle and Sql Server: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/DB_connection.R
  * NOTE1: When using R to connect to Oracle, as Oracle SQL query requires you to use double quote for Alias, not single quites. Meanwhile, in R `dbGetQuery()` you have to use double quotes for the whole query. Then you can just use `\` in fornt of each double quote for Oracle query. For example, `dbGetQuery(con, "select col as \"Column1\" from my_table")`
  * NOTE2: When using R to connect to SQL Server using RODBC, the limitation is each handler points to 1 database, therefore, you cannot join tables from multiple databases in 1 SQL Query in R. But! You can use R `merge` function to do Nature Join (special case of inner join), Left Join, Right Join and Full Outer Join. When I was running large amount of data, R even do joins faster than SQL Server!
  * NOTE3: Because of the limitation of RODBC mentioned in NOTE2 above, sometimes before merging, the existing 2 pieces of data may occupy large memory and there will be out of memory error when you try to join data. When this happen, try this `options(java.parameters = "-Xmx3g")`, this means change the R memory into 3 GB
 
* Simple Example to do joins in R for SQL Server query: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/R_SQLServer_multiDB_join.R
  * magrittr, a method replces R nexted functions: https://github.com/hanhanwu/magrittr

* <b>Challenges of Using R, and Compare with MapReduce </b>
  * Paper Source: http://shivaram.org/publications/presto-hotcloud12.pdf
  * R is primarily used as a single threaded, single machine installation. R is not scalable nor does it support incremental processing.
  * Scaling R to run on a cluster has its challenges. Unlike MapReduce, Spark and others, where only one record is addressed at a time, the ease of array-based programming is due to a global view of data. R programs maintain the structure of data by mapping data to arrays and manipulating them. For example, graphs are represented as adjacency matrices and outgoing edges of a vertex are obtained from the corresponding row.
  * Most real-world datasets are sparse.  Without careful task assignment performance can suffer from load imbalance: certain tasks may process partitions containing many non-zero elements and end up slowing down the whole system.
  * In incremental processing, if a programmer writes y = f(x), then y is recomputed automatically whenever x changes. Supporting incremental updates is also challenging as array partitions which were previously sparse may become dense and vice-versa. 


********************************************************

CLOUD PLATFORM MACHINE LEARNING

* AWS

  * AWS first step: http://www.analyticsvidhya.com/blog/2016/05/complete-tutorial-work-big-data-amazon-web-services-aws/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * Hanhan's AWS work: https://github.com/hanhanwu/Hanhan_AWS
  * Install R Studio in AWS: https://www.analyticsvidhya.com/blog/2015/07/guide-quickly-learn-cloud-computing-programming/?utm_content=bufferae4c7&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer


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
  * Using R in Azure Data Lake (only basic filesystem operations, but I think it's interesting at this time): https://blogs.msdn.microsoft.com/microsoftrservertigerteam/2017/03/14/using-r-to-perform-filesystem-operations-on-azure-data-lake-store/


* Spark

  * Spark Cloud User Guide: https://docs.databricks.com/user-guide/tables.html
  * Spark Advanced Analysis: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/Mastering-Advanced-Analytics-With-Apache-Spark.pdf
  * Spark 2.0 live presentation: https://www.brighttalk.com/webcast/12891/202021?utm_campaign=communication_reminder_starting_now_registrants&utm_medium=email&utm_source=brighttalk-transact&utm_content=button
  * Spark 2.0 code documentation (Spark 2.0 preview is on Spark Community Edition now!): https://people.apache.org/~pwendell/spark-nightly/spark-master-docs/latest/sql-programming-guide.html
  * Spark R live tutorial: https://www.brighttalk.com/webcast/12891/202705?utm_campaign=communication_reminder_24hr_registrants&utm_medium=email&utm_source=brighttalk-transact&utm_content=title
  * Deep Dive in Spark Memory Management: http://go.databricks.com/e1t/c/*W8FW5K19b2C3QW67F6YB4n5y0q0/*W2cBSXl3Bz_cwW3RPRtC1S20zS0/5/f18dQhb0SbTM8Y9Wm6W7my7NM4T_wCFVd773C2zGqr0Msd2JqXD6prW39Dr-S96dVWSW1n3m4t5DrCyhW4vgKM197BYcrVcjcsf3KBRrrVbY4by6bW3GrW50SXS06PkKHkW2yj0SQ51frwmW6bVCNc1nWTVGW6PVKcr6N3nBLW8xGZPk6b0-bSW50SM1Y74yXDCW5ZpycS57-ZBTW83C5JP6rY-3sW2KQ2YY1Gk5dXN2KgH_18JJbJW8cCFLm3X0g0yN6bp1cMbq5fxN4TKc2lSmND-W4FlHY21n-6V3W1qRbjk2zHM3BW735zWg5D3x8fW96Cy694XqCsJW1mj2g25PPw2SM3WjxTQBz10W6R9m946Qy77mW2gF3pQ3sC0w-VN-ByV6NMg3-W3KxvwR8RLQRpMbWF_jfQckVW1bBvW33_5Xh9W23b7dn1NFfl4W8P4wwZ8RhNw8V2Czct47GttxW4b9Kky5qdN3FW7jqmY_7rJyr7W4pGwxr728tL1W44pRdR4kr26zW7x91Q96n6qz-w3QzRXbcvbf38bCJG02
  * 2016 survey results, we somdeties use Spark survey to reflect the trends in this area: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/2016_Spark_Survey.pdf


********************************************************

VISUALIZATION

* Hanhan's d3 practice: https://github.com/hanhanwu/Hanhan_Data_Visualization
* Tableau Data Visualization White paper: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/whitepaper_visual-analysis-guidebook_0.pdf
* Detailed R Visualization methods (I like this one!): http://www.analyticsvidhya.com/blog/2016/03/questions-ggplot2-package-r/?utm_content=buffer18a4b&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* Tableau learning resources: https://www.analyticsvidhya.com/learning-paths-data-science-business-analytics-business-intelligence-big-data/tableau-learning-path/
* d3 resources (too basic), in fact you can simply use JS Bin and embed d3 library in javascript with only 1 line: https://www.analyticsvidhya.com/learning-paths-data-science-business-analytics-business-intelligence-big-data/newbie-d3-js-expert-complete-path-create-interactive-visualization-d3-js/?utm_content=bufferf83d2&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* R Shiny - creating interactive visualization: https://www.analyticsvidhya.com/blog/2016/10/creating-interactive-data-visualization-using-shiny-app-in-r-with-examples/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Plotly (interactive visualization methods, can be used with multiple data science languages and D3, many of the samples here can be done in Spark Cluster): https://www.analyticsvidhya.com/blog/2017/01/beginners-guide-to-create-beautiful-interactive-data-visualizations-using-plotly-in-r-and-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

* What to note when using PowerBI (free version)
  * A friend said PowerBI performs faster than Tableau10 when the data set is large, and there are many online libraries to download. So, it's still worthy to use PowerBI for data visualization. It's just as other MSFT products, never make your life easier although it has many functions looks cool. So, need to write down some notes when using it.
  * When using the free version, and want to create interactive vusualization that contains multiple charts, with multiple dataset, PowerBI desktop has more flexibility. But if we want to publish it to PowerBI dashboard, we could just publish the saved visualization file from Desktop
  * When the dataset for the visualization has chanhed, if the data structure has not been changed, click `Refresh` through PowerBI Desktop, it maybe able to update. Sometimes, when you only update several datasets instead of update them all, you may not be able to refresh, since the relationship between tables may put limitation on data refresh. When this problem happened, try to check the relationship between tables, and when updating the datasets, make sure these relationship won't be broken...
  * When you want to generate an url and let people see. There are 2 ways. One way is, on PowerBI Dashboard, Click Publish, then Click Share, the generated url can be seen by everyone. The other way is to Right Click the name of the dashboard you want to share, then grant the audience access by typying their emails. Click Access, the generated url can only be shared by these people. One thing to note is, when you are granting the access to the audience, those who with only emails have not set up PowerBI, those who with PowerBI Account Name have set up the PowerBI.
  * It is more convenient if your audience have installed PowerBI Mobile App, in this way, without sending them url but just grant them the access to your dashboard, they can see it through their mobile devides immediately.
 
* PowerBI Pro
  * The Gateway demo part: https://www.youtube.com/watch?v=zIu3YXPGRBc&feature=youtu.be
 
* Data Visualization with QlikView: https://www.analyticsvidhya.com/blog/2015/12/10-tips-tricks-data-visualization-qlikview/?utm_content=buffera215f&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer



********************************************************

DEEP LEARNING

* A detailed deep learning example: https://www.analyticsvidhya.com/blog/2016/11/tryst-with-deep-learning-in-international-data-science-game-2016/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Color the photo with NN (this article is very interesting and it reminds me of the Facebook upcoming feature, which allows you to choose an art style for your photo): https://www.analyticsvidhya.com/blog/2016/11/creating-an-artificial-artist-color-your-photos-using-neural-networks/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* NN with Theano: https://www.analyticsvidhya.com/blog/2016/11/fine-tuning-a-keras-model-using-theano-trained-neural-network-introduction-to-transfer-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* How to build Deep Learning Workstation within $5000: https://www.analyticsvidhya.com/blog/2016/11/building-a-machine-learning-deep-learning-workstation-for-under-5000/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


********************************************************

Industry Data Analysis/Machine Learning Tools

* IBM SPSS
  * IBM SPSS Modeler Cookbook (only for SFU students): http://proquest.safaribooksonline.com.proxy.lib.sfu.ca/9781849685467


********************************************************

Statistical Methods

* Hypothesis Tests and relative statistics knowledge: http://www.analyticsvidhya.com/blog/2015/09/hypothesis-testing-explained/
* z-score and p-value table: http://www.stat.ufl.edu/~athienit/Tables/Ztable.pdf
* More about Correlation: http://www.analyticsvidhya.com/blog/2015/06/correlation-common-questions/?utm_content=buffer28126&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* Methods used in Psychological Analysis
  * Cronbach's Alpha, measuring the internal consistency: http://data.library.virginia.edu/using-and-interpreting-cronbachs-alpha/
  * R psychological analysis package 1: https://cran.r-project.org/web/packages/psy/psy.pdf
  * R psychological analysis package 2: https://cran.r-project.org/web/packages/psych/psych.pdf
  * A detailed example of R psychological research: http://personality-project.org/r/r.guide.html#alpha
  * Methods to measure reliability (Cronbach's Alpha is most frequently used): http://www.socialresearchmethods.net/kb/reltypes.php
  * Test Validity: https://en.wikipedia.org/wiki/Test_validity
  * AV Casino - Probability (Need Registration): https://www.analyticsvidhya.com/blog/2016/08/launch-of-av-casino-an-introduction-to-probability/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * Probability in Mahcine Learning: https://github.com/hanhanwu/readings/blob/master/probability%20in%20machine%20learning.pdf


********************************************************

Terminology Wiki

* Multicollinearity: https://en.wikipedia.org/wiki/Multicollinearity
* Autocorrelation: https://en.wikipedia.org/wiki/Autocorrelation#Regression_analysis
* Heteroscedasticity: https://en.wikipedia.org/wiki/Heteroscedasticity
* Bias: how much on an average are the predicted values different from the actual value
* Variance: how different will the predictions of the model be at the same point if different samples are taken from the same population
* Ensemble methods (bagging, boosting, stacking) are used to keep a balance between bias and variance


********************************************************

Data Analysis Tricks and Tips

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


-- TIME SERIES

* 3 Winners deal with mini time series challenge (very interesting, especially after seeing the champion's code..): http://www.analyticsvidhya.com/blog/2016/06/winners-mini-datahack-time-series-approach-codes-solutions/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Time Series Modeling
  * Tutorial: http://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/?utm_content=buffer529c5&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
  * My R code (more complete): https://github.com/hanhanwu/Hanhan_Data_Science_Practice/blob/master/time_series_predition.R


-- Segmentation

* Segmentation, what to note: https://www.analyticsvidhya.com/blog/2016/02/guide-build-predictive-models-segmentation/?utm_content=bufferb3404&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 

-- Use Clustering with Supervised Learning
 
* Detailed article: https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-clustering-and-different-methods-of-clustering/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* The above article is a real good one, besides some details I didn't learn from school, one thing I really like, it uses clustering for the data first, then added a new feature in the dataset which is the cluster numbers of the data, then do supervised learning. This is new but interesting to me!


********************************************************

Machine Learning Experiences

* Experiences Notes: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/Experiences.md
* Public datasets: https://www.analyticsvidhya.com/blog/2016/11/25-websites-to-find-datasets-for-data-science-projects/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


********************************************************

CROWD SOURCING

* Crowd sourcing for Taxonomy: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/ChiltonCascadeCHI2013.pdf
* Improved Crowd sourcing for Taxonomy: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/hcomp13.pdf
* CrowdFlower, the crowd sourcing platform (not free): https://www.crowdflower.com/
* Amazon MTurk (not free): https://requester.mturk.com/create/sentiment/about
* One day, when I become rich and popular enough, I should fund a Passion Group, organizing qualified people to do crowd-sourcing volunteer work, because we all have passion


********************************************************

GOOD TO READ

* Causality and Correlation: http://www.analyticsvidhya.com/blog/2015/06/establish-causality-events/
* Why NoSQL: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/IBM%20Q1%20Cloud%20Data%20Services%20Asset%232%20-%20Why%20NoSQL%20ov38975.pdf
* Data Science Life Cycle: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/IBM%20Q1%20Technical%20Marketing%20ASSET2%20-%20Data%20Science%20Methodology-Best%20Practices%20for%20Successful%20Implementations%20ov37176.pdf
* Machine Learning Algorithms Overview with R/Pyhton code: http://www.analyticsvidhya.com/blog/2015/08/common-machine-learning-algorithms/
* Genius in data science, they really make our life easier: http://www.analyticsvidhya.com/blog/2016/05/infographic-16-genius-minds-inventions-data-science-easier/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* 8 reasons for failed model building: http://www.analyticsvidhya.com/blog/2016/05/8-reasons-analytics-machine-learning-models-fail-deployed/?utm_content=buffer13396&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* Basic Python Visualization: http://www.analyticsvidhya.com/blog/2015/05/data-visualization-python/?utm_content=buffer4e100&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* DataRobot Automated Machine Learning: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/How_to_Automate_Machine_Learning-FINAL-print.pdf
* Basic Data Integration with HDFS and DMX-h: http://www.analyticsvidhya.com/blog/2016/06/started-big-data-integration-hdfs-dmexpress/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Data exploraton in <b>SAS</b>: http://www.analyticsvidhya.com/blog/2015/04/data-exploration-sas-data-step-proc-sql/
* <b>SAS Macros</b> for faster data manipulation: http://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-sas-macros-faster-data-manipulation/?utm_content=buffer252dc&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* Basic Data Exploration with Python pandas, numpy (Spark provides functions for all of these): http://www.analyticsvidhya.com/blog/2015/04/comprehensive-guide-data-exploration-sas-using-python-numpy-scipy-matplotlib-pandas/?utm_content=buffer01cc9&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* Fast and simple way to build predictive model: http://www.analyticsvidhya.com/blog/2015/09/perfect-build-predictive-model-10-minutes/
* Python Learning Path (I haven't tried deeplearning.net yet): http://www.analyticsvidhya.com/learning-paths-data-science-business-analytics-business-intelligence-big-data/learning-path-data-science-python/
* Improve Accuracy of Machine Learning Model: http://www.analyticsvidhya.com/blog/2015/12/improve-machine-learning-results/?utm_content=bufferf5391&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* Bayesian Statistics for Beginners: http://www.analyticsvidhya.com/blog/2016/06/bayesian-statistics-beginners-simple-english/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Operations Analytics: http://www.analyticsvidhya.com/blog/2016/06/operations-analytics-case-study-level-hard/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Product Price Optimization, without using any built-in libraries: http://www.analyticsvidhya.com/blog/2016/07/solving-case-study-optimize-products-price-online-vendor-level-hard/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Excel tips: http://www.analyticsvidhya.com/blog/2015/11/excel-tips-tricks-data-analysis/?utm_content=buffer5fd6a&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* Time Series Tutorial: http://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/?utm_content=buffer529c5&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* Gracefully shutdown Spark Streaming: https://www.linkedin.com/pulse/apache-spark-streaming-how-do-graceful-shutdown-chandan-prakash?trk=hp-feed-article-title-like
* For beginners, major questions that data science could solve: https://blogs.technet.microsoft.com/machinelearning/2015/08/27/what-types-of-questions-can-data-science-answer/
* Data Science with Command Line (Although I will think R can do all these, it's still an interesting article that it covers data exploration, visualization and modeling examples with command line): https://www.analyticsvidhya.com/blog/2016/08/tutorial-data-science-command-line-scikit-learn/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Core concepts of Neural Network (a very good resource for learning or review): https://www.analyticsvidhya.com/blog/2016/08/evolution-core-concepts-deep-learning-neural-networks/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
 
 -- In this article, when they were talking about concepts such as Activation Function, Gradient Descent, Cost Function, they give several methdos for each and this is very helpful, meanwhile, I have leanred deeper about BB through the concept of Momentum, Softmax, Dropout and Techniques dealing with class imbalance, very helpful, it is my first time to learn deeper about these

* Learning from winners, the power of feature engineering (does it also tell me, I should apply for jobs earlier): https://www.analyticsvidhya.com/blog/2016/08/winners-approach-smart-recruits/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Active Clean - An Interactive Data Cleaning Method: https://github.com/hanhanwu/Hanhan_Data_Science_Resources/blob/master/vldb2016-activeclean.pdf
* Video lectures in data science area (interesting): https://www.analyticsvidhya.com/blog/2015/07/top-youtube-videos-machine-learning-neural-network-deep-learning/?utm_content=buffer42327&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* CheatSheet, popular ML algorithms, dmension reduction in R and Python: https://www.analyticsvidhya.com/blog/2015/09/full-cheatsheet-machine-learning-algorithms/?utm_content=buffere30ff&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* Potential APIs (visual, auto/real time/deeper analysis/prediction, intelligent assistence, NLP, machine learning/big data platform): https://www.analyticsvidhya.com/blog/2016/09/what-should-you-learn-from-the-incredible-success-of-ai-startups/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* A big data video, DataHub: https://youtu.be/RnoWX2Csuv8
* DataHub related to the above video: https://datahub.csail.mit.edu/www/
* How to avoid common analysis mistakes: https://www.analyticsvidhya.com/blog/2013/06/common-mistakes-analysis-avoid-them/?utm_content=bufferbc729&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
 
 -- From the above article, I have made the summary that I think needs to keep in mind:

* When drawing inferences from the data, check distributions and outliers first, and see whether you could use mean/mode or median.
* Comparing different segment/cluster of data, compare Pre & Post situations.
* <b>Extrapolation</b> - the process of estimating, beyond the original observation range, the value of a variable on the basis of its relationship with another variable.
* <b>Confidence Interval</b> - a range of values so defined that there is a specified probability that the value of a parameter lies within it.
* When doing extrapolation, always plot the confidence interval to the values to extrapolate, it's safer when it reaches to at least 90% confidence interval.
* When the model has been extended to the population without past, check distribution of key features, if there is not too much change, it's safe, otherwise, changes of the model maybe needed.
* Correlation is correlation, has nothing to do with causation.

* Shelf space optimization with linear programing: https://www.analyticsvidhya.com/blog/2016/09/a-beginners-guide-to-shelf-space-optimization-using-linear-programming/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Compared with the above article, here is how Amazon arranges its warehoue, and I really like this idea: http://www.businessinsider.com/inside-amazon-warehouse-2016-8

* Implement NN with <b>TensorFlow</b> [lower level library], image recognition example: https://www.analyticsvidhya.com/blog/2016/10/an-introduction-to-implementing-neural-networks-using-tensorflow/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Imagine recognition, using NN with <b>Keras</b> [higher level library]: https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/
 
* Data Science books in R/Python for beginners (after checking these books in school library, I really think they are for beinners, and some are too basic, not sure why so many people recommend these books....): https://www.analyticsvidhya.com/blog/2016/10/18-new-must-read-books-for-data-scientists-on-r-and-python/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

* Emotion Intelligence with Visual and Spark (it's very interesting to know that in their work, they are also trying to predict what kind of users will become the failure of data collection, this may improve the data management): http://go.databricks.com/videos/spark-summit-eu-2016/scalable-emotion-intelligence-realeyes?utm_campaign=Spark%20Summit%20EU%202016&utm_content=41933170&utm_medium=social&utm_source=facebook

* A good reading about data APIs and some cool projects used these APIs (I'm especially interested in IBM personal insights): https://www.analyticsvidhya.com/blog/2016/11/an-introduction-to-apis-application-programming-interfaces-5-apis-a-data-scientist-must-know/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


*****************************

DATA SCIENCE INTERVIEW PREPARATION

* Machine Learning Videos (The Lessones learned from Pinterest is good!): https://www.analyticsvidhya.com/blog/2016/10/16-new-must-watch-tutorials-courses-on-machine-learning/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* 20 "challenging" interview puzzles (who came up with these question...): http://www.analyticsvidhya.com/blog/2016/07/20-challenging-job-interview-puzzles-which-every-analyst-solve-atleast/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29 
* R skillset test: https://www.analyticsvidhya.com/blog/2016/08/full-solution-skill-test-on-r-for-data-science/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Statistics test: https://www.analyticsvidhya.com/blog/2016/08/solutions-for-skilltest-in-statistics-revealed/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Statistics test II: https://www.analyticsvidhya.com/blog/2016/09/skilltest-statistics-ii-solutions/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Machine Learning skillests test: https://www.analyticsvidhya.com/blog/2016/11/solution-for-skilltest-machine-learning-revealed/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* z-score to p-value calculator: http://www.socscistatistics.com/pvalues/normaldistribution.aspx
* A convenient t-table: http://www.sjsu.edu/faculty/gerstman/StatPrimer/t-table.pdf
* How to read the above t-table for hypothesis test: https://www.youtube.com/watch?v=omY1pgKbr3I
* 40 Data Science Internview QUestions: https://www.analyticsvidhya.com/blog/2016/09/40-interview-questions-asked-at-startups-in-machine-learning-data-science/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* Python data science skill test: https://www.analyticsvidhya.com/blog/2016/09/solutions-data-science-in-python-skilltest/?utm_content=buffer8e6da&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* <b>Questions to ask interviewers</b> (these suggestions are helpful, data science used in analytics jobs are still new, it is difficult to find the rigth team form their job description, digging deeper during the interview may help. I'm also thinking hot to find the manager and colleagues' working style, and see whether I can collaborate with them better...): https://www.analyticsvidhya.com/blog/2013/09/analytics-job-5-questions/?utm_content=buffer88caa&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer


***************************************

LEARING FROM THE OTHERS' EXPERIENCES

* suggestions about analytics work (it looks useful, I just feel howcome these people in India are doing lots of data analytics work with machine learning knowledge, but in Vancouver or even in Canada, everything looks so out of dated, slow-paced. When can I find a satisfied job?): https://www.analyticsvidhya.com/blog/2013/07/analytics-rockstar/?utm_content=buffer3655f&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
* The feature engineering here has some good points I could try: https://www.analyticsvidhya.com/blog/2016/10/winners-approach-codes-from-knocktober-xgboost-dominates/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

* Suggestions for data science work: https://www.analyticsvidhya.com/blog/2015/11/exclusive-interview-srk-sr-data-scientist-kaggle-rank-25/
* Suggestions from a top data scientist (I really like this one): https://www.analyticsvidhya.com/blog/2013/11/interview-top-data-scientist-kaggler-mr-steve-donoho/
* winner strategies: https://www.analyticsvidhya.com/blog/2016/10/winning-strategies-for-ml-competitions-from-past-winners/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * Data Exploration
  * Feature Engineering (Feature Selection, Feature Transformaton, Feature Interaction and Feature Creation)
  * Validation to prevent from overfitting
  * Try Feature Selection with Cross Validation
  * Methods like R `findCorrelation()`, PCA could help feature selection when there is no label (dependent variable); methods like GBM, XGBoost, Random Forest, <b>R Boruta</b> (a very simple feature selection method) and PLS could tell feature importance when there is a label (dependent variable)
  * Model Ensembling!
  * <b>Sometimes can create derived dependent variable for prediction</b>
  * Review my evaluation metrics notes: https://github.com/hanhanwu/readings/blob/master/Evaluation_Metrics_Reading_Notes.pdf

* Add external view for KPI: https://www.linkedin.com/pulse/one-important-thing-missing-from-most-kpi-dashboards-bernard-marr?trk=hp-feed-article-title-like

* Tuning Random Forest Params - Python
  * https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/?utm_content=buffer94fa3&utm_medium=social&utm_source=facebook.com&utm_campaign=buffer
  * The writer found <b>an ensemble with multiple models of different random states</b> and <b>all optimum parameters</b> sometime performs better than individual random state.
  * oob_score is similar to leave one out validation but faster.

* https://www.analyticsvidhya.com/blog/2016/10/winners-solution-from-the-super-competitive-the-ultimate-student-hunt/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
* In the above article, I have made these summary:
  * xgboost is a real good one for time series prediction or normal prediction
  * xgboost will show the iportance of features too, which is hepful
  * feature engineering is very important
  * one-hot encoding is helpful too
  * understanding missing data can be helpful too
 
* Suggestions from a top data scientist: https://www.analyticsvidhya.com/blog/2016/10/exclusive-interview-ama-with-data-scientist-rohan-rao-analytics-vidhya-rank-4/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

* Learning from winners, the power of feature engineering (does it also tell me, I should apply for jobs earlier): https://www.analyticsvidhya.com/blog/2016/08/winners-approach-smart-recruits/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
  * In this article, when they were talking about concepts such as Activation Function, Gradient Descent, Cost Function, they give several methdos for each and this is very helpful, meanwhile, I have leanred deeper about BB through the concept of Momentum, Softmax, Dropout and Techniques dealing with class imbalance, very helpful, it is my first time to learn deeper about these
 
* 3 Winners deal with mini time series challenge (very interesting, especially after seeing the champion's code..): http://www.analyticsvidhya.com/blog/2016/06/winners-mini-datahack-time-series-approach-codes-solutions/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29


***************************************************************

OTHER

* They even have Data Sciecne games to break ice/Monday blues... I don't have Monday blues, but it's interesting to see these: https://www.analyticsvidhya.com/blog/2016/11/8-interesting-data-science-games-to-break-the-ice-monday-blues/?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29

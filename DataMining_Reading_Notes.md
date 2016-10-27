
* Data Types Teerminology
 * Nominal data, categorical
 * Binary data, 0 & 1 categorical
 * Ordinal data, values with meaningful order or ranking. (eg. small, medium, large; dissatisfied, neural, satisfied, very satisfied)
 * <b>The central tendency of ordinal data can be defined by mode or median, no mean for this data type</b>
 * Numerical data
 

* Central Tendency
 * Positive skewed: x of mode < x of median
 * Negative skewed: x of mode > x of median
 

* Measures of Data Dispersion
 * Range: min ~ max
 * Quantile, dividing into equal size of consecutive sets
 * Percentile, 100-quantiles
 * Q1, First Quantile, lowest 25%; Q3, Third Quantile, lowest 75%; Q2, Second Quantile = median, 50%
 * IQR, Interquartile Range, IQR = Q3 - Q1
 * <b>Rule of thumb for identifying suspected outliers is to single out values falling at least 1.5*IQR above Q3 or below Q1</b>
 * Five Number Summary: min, Q1, Q2 (median), Q3, max
 * VISUALIZATION Boxplot - outliers:
   1. The <b>median</b> is marked by the line in boxplot
   2. The <b>length of the box</b> is IQR, since the ends of it are at quantiles
   3. <b>2 lines (whiskers) out of the box</b> reaches to min, max
   4. If we <b>extend whiskers to 1.5*IQR higher than Q3 and lower than Q1</b>, those individual dots will be the potential outliers
 * Standard Deviation: a low standard deviation means the data tends to be close to the mean; a high standard deviation indicates that the data spread out over a large range of values
 * VISUALIZATION Scatter Plot, Scatter Plot Matrix - correlation: when y increases with X, positive correlation; y decreases when X increases, negative correlation
 * VISUALIZATION Histogram - distribution
 * VISUALIZATION Pixel Visualization - reflect trends of multiple features at the same time: smaller the values, lighter the shading


* Dealing with Missing Data
 * When you are using central tendency (mean or median), if the distribution of the data is skewed, median is better; 
 for normal (symmetric) data, mean is better.
 * A popular way is to predict missing data with methods such as decison trees, bayesian inference and regression
 
 
* Dealing with Noisy Data
  * <b>Binning</b>: Put data into bins and perform LOCAL SMOOTHING. Smoothing by bin means; Smoothing by bin medians, smoothing by bin boundries (min, max). Bins maybe equal width, where the interval range of values in each bin is constant.
  * Binning can also be used in data resudtion for data that has too many distinct values
  * <b>Regression</b>: Conform data values to a function, to find the best line fits the variables, so that n-1 variables can predict the nth variable.
  * <b>Clustering</b>: Outliers detection


* Discrepancy Detection
 * Is the variable symmetric or skewed? Do all Values fall within acceptable range? Are there any KNOWN dependency between variables?
 * Values more than 2 standard deiation away from the mean can be potential outliers.
 * Deal with data inconssitency/errors
 * Standardize data format
 * While doing data transformation to improve data quality, some transformation may bring in more data discrepancy, some may only be solved after others have been solved
 * Publick Tools that integrates discrepency detectio and data transformation, Potter's Wheel: http://control.cs.berkeley.edu/abc/
 * Potter's Wheel paper: http://control.cs.berkeley.edu/pwheel-vldb.pdf
 
 
* Dealing with Data Redundancy
  * Correlation Analysis - for norminal data, chi-square test
  * Correlation Analysis - for numerical data, correlation co-efficient and covariance
  * chi-square hypothesis: variables are independent (no correlation between them), if this hypothesis can be rejected, the variables are correlated
  * Cells that contributes the most to chi-square value are those for which the actual count is very different from that expected
  * Correlation Coefficient or Covariance (similar measures): > 0, positive correlated; = 0, independent; < 0 negative correlated
  
  
* Data Reduction
  * <b>Dimensional Reduction</b>: reducing number of variables under consideration. Wavelet transforms, Principal Component Analysis (PCA)
  * <b>Numerosity Reduction</b>: replace original data volume with smaller forms, parametric and non-parametric methods
  * <b>Data Compression</b>: lossless or lossy, limited data manipulation
  * Wavelet Transform (DWT) - tends to save more space and provide more accurate approximation of the orginal data. <b>Works well on sparse or skewed data and on data with ordered variables</b>
  * PCA - can be applied to both ordered and unordered variables, can handle sparse and skewed data. But for multidimensional data with more than 2 domwnsions, need to reduce the problem into 2 dimensions first
  * PCA vs. DWT: PCA works better on sparse data while DWT works on high domensional data
  * Attribute subset selection - find a minimum set of attributes such that the resulting probability distribution of the data classees is as close as possible to the original distribution using all the data. <b>Heuristic methods</b> are popular.
  * Regression, Log-Linear Models
  * Histogram, simple and highly efficient for sparse/dense data, highly skewed data, uniformed data. Equal width; Equal frequency
  * Clustering
  * Sampling
  
* <b>Data Transformation Strategy Summary</b>
  1. Remove Noisy Data
  2. Feature Engineering
  3. Feature Aggregation
  4. Data Normalization
  5. Discretization - raw values of numerical feature are replaced by interval labels (eg. 1-10, 11-20) or conceptual labels (youth, adult, senior)
  6. Concept hierarchy generation for nominal data
  

* <b>Data Preprocessing Strategy Sumary</b>
  1. Check Data Quality
  2. Initial Data Cleaning: missing data, deal with outliers/data errors, etc
  3. Data Integration
  4. Data Reduction
  5. Data Transformation
  
  
* Data Normalization
 * Attempts to give features equal weight. Especially help NN and distance measurements based algorithms such as clustering, KNN
 * min-max normaliation
 * z-score normalization: works when min, max are unknown or when there are outliers that dominate min-max normalization
 * decimal scaling
 
 
* Discretization
 * Clustering, popular
 * Decision trees with Entropy, top-down splitting strategy
 * Measure of Correlation, ChiMerge, bottom-up splitting strategy


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

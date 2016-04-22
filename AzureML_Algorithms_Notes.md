Record what does the algorithms in AzureML do.
Before using AzureML algorithms, it is better to know the background algorithms behind those modules.


-- One-vs-All Multiclass 

* It can use any binary classifier as an input to solve a multi-class classification problem, 
based on one-vs-all method. For example, you can use a binary classification module, 
Two-Class Support Vector Machine, and connect it to the One-vs-All Multiclass module.
* Background Algorithm: https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest


-- Desion Forest, Decision Jungle, Boosted Decision Trees, Fast forest quantile regression

* The average of this "decision forest" is a tree that avoids overfitting. 
* Decision forests can use a lot of memory. Decision jungles are a variant that consumes less memory at the expense of a slightly longer training time.
* Boosted decision trees avoid overfitting by limiting how many times they can subdivide and how few data points are allowed in each region. The algorithm constructs a sequence of trees, each of which learns to compensate for the error left by the tree before. The result is a very accurate learner that tends to use a lot of memory. 
* Fast forest quantile regression is a variation of decision trees for the special case where you want to know not only the typical (median) value of the data within a region, but also its distribution in the form of quantiles. https://msdn.microsoft.com/library/azure/dn913093.aspx


-- Neural Network

* Neural networks are brain-inspired learning algorithms covering multiclass, two-class, and regression problems.
* Neural networks within Azure Machine Learning are all of the form of directed acyclic graphs. That means that input features are passed forward (never backward) through a sequence of layers before being turned into outputs.
* Two-class Averaged Perceptron is neural networks' answer to skyrocketing training times. It uses a network structure that gives linear class boundaries. It is almost primitive by today's standards, but it has a long history of working robustly and is small enough to learn quickly.
* Specify your own NN: https://azure.microsoft.com/en-us/documentation/articles/machine-learning-azure-ml-netsharp-reference-guide/


-- SVM

* Two-class SVM separates 2 classes with a straight line only. (In SVM-speak, it uses a linear kernel.) Because it makes this linear approximation, it is able to run fairly quickly, with less overfitting than most other algorithms, in addition to requiring only a modest amount of memory.
* Two-class locally deep SVM is a non-linear variant of SVM that retains most of the speed and memory efficiency of the linear version. It is ideal for cases where the linear approach doesn't give accurate enough answers.
* One-class SVM draws a boundary that tightly outlines the entire data set. It is useful for anomaly detection. Any new data points that fall far outside that boundary are unusual enough to be noteworthy.   [MSR Product]


-- Bayesian methods

* Avoid overfitting, but by making some assumptions beforehand about the likely distribution of the answer, and they have very few parameters.
* Two-class Bayes' Point Machine, for classification.    [MSR Product]
* Bayesian Linear Regression, for regression


-- Specialized Algorithms

* Rank Prediction - Ordinal Regression
* Count Prediction - Poisson Regression
* Anomaly Detection - Principal Components Analysis (PCA), One-class SVM

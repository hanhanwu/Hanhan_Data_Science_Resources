Record what does the algorithms in AzureML do.
Before using AzureML algorithms, it is better to know the background algorithms behind those modules.


-- One-vs-All Multiclass 

* It can use any binary classifier as an input to solve a multi-class classification problem, 
based on one-vs-all method. For example, you can use a binary classification module, 
Two-Class Support Vector Machine, and connect it to the One-vs-All Multiclass module.
* Background Algorithm: https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest

# Random Forest

## Introduction

This is a project by Tristan Miller ([@tristanlmiller](https://github.com/tristanlmiller/)) from 2018.  I implemented my own version of a random forest of decision trees *without* scikit-learn, using only pandas.  Originally this was intended to address a specific binary classification problem.  However, I cannot share the data, so I'm just sharing the code I wrote, as a coding demonstration.

## What is a random forest?

Decision trees are a common method to classify multidimensional data.  Each interior node of the tree contains a particular condition; if a data point fulfills the condition then it is sent to the right branch, otherwise it is sent to the left.  When the data point reaches a leaf of the tree, the leaf determines how the data is classified.

Using labeled data, you can train decision trees to predict the labels of unlabeled data.  However, decision trees tend to overfit the training data.  A common fix is to train many decision trees, and have the trees vote on how to classify data.  To simulate the variance of real data, each decision tree is given a slightly different training set, each one selected randomly (with replacement) from the original training set.  Furthermore, each tree is restricted to making decisions based only on a small sample of the data's features.  This is called a random forest.

## How to use this code

The module *decision_trees.py* contains a class called *RandomForest* which will create, train, and validate a random forest, and use it to predict the labels of any data set.  This implementation is limited to binary classification (i.e. the labels must be 1s and 0s), and numerical features (i.e. no strings).

To initialize a *RandomForest*, the user should pass it a list of options:

*RandomForest(feature_cols,label_col,iterations,num_trees,num_features,target_runtime,fix_iter=True,weight=0.5)*

- *feature_cols* is a slice with the indices of the columns of the features.
- *label_col* is the index of the column with the labels.
- *iterations* limits the number of nodes in each tree if *fix_iter* is true (see below).
- *num_trees* limits the number of trees if *fix_iter* is false (see below).
- *num_features* is the number of dimensions that each decision tree is allowed to look at.
- *target_runtime* is the approximate length of time (in minutes) you are willing to let the trees train.
- If *fix_iter* is true (default) then each tree is grown with a fixed number of nodes, and more trees will be grown until time runs out.  If *fix_iter* is false, then a fixed number of trees will be grown with enough nodes to reach the target runtime.
- *weight* is the weight assigned to correctly predicting labels whose value is 1.  *weight* ranges between 0 and 1.

To train the forest, use *RandomForest.train(training_data)*.  All data must be in a pandas DataFrame.  To evaluate the forest on any data set, use *RandomForest.confusion(evaluation_data)* to return the confusion matrix.  Use *RandomForest.evaluate(evaluation_data)* to return the accuracy of the forest.  *RandomForest.train_valid(training_data,evaluation_data)* will train the random forest, and then print the accuracy.  Finally, *RandomForest.predict_labels(data)* will return the predictions of the labels on any data set.

## Details of implementation

*decision_trees.py* contains three classes: *TreeNode*, *DecisionTree* and *RandomForest*.

An instance of *TreeNode* is a single node of a decision tree.  There are two types of *TreeNode*s: leaves and interior nodes.  Each leaf contains a single guess, either a 0 or 1.  Each interior node contains pointers to two children, and a partition on a single feature.  Each leaf in a decision tree is also assigned a unique ID from 0 to N-1, where N is the number of leaves.

*TreeNode* contains a procedure for leaves to produce new children.  The leaf looks at all the available features, and chooses the partition that results in the greatest classification accuracy of the training data.  If there is no partition that improves on the current accuracy, then a random feature is selected and partitioned along the median value.  

*DecisionTree* is a wrapper class for each decision tree.  Each instance contains a pointer to the root *TreeNode*, and additional information needed for training the tree.  Although *DecisionTree* does not contain a copy of the training data itself, I found it efficient for it to track how the training data is currently being classified.

At each iteration of the training algorithm, *DecisionTree* selects the leaf that has the most incorrectly classified data points.  This leaf is directed to produce new children.  Training terminates if all data is correctly classified.

Finally, *RandomForest* contains a list of *DecisionTree*, and various user-facing functions.  Each *DecisionTree* is given a random sample of the training data (chosen with replacement), and a random subset of the features.

## Potential areas for improvement

In a comparison to sklearn's RandomForestClassifier, my algorithm does not perform as well, and in particular it's hundreds of times slower in terms of computation time.  I do not know what exactly sklearn does to improve its efficiency, although I suspect that it's working with vectorized computation.

A few possible areas where I think my training algorithm could be made more efficient. First, there is the algorithm used to choose the best partition for the data.  The partition is chosen by a greedy algorithm, and maybe there's something better.  Second, rather than generating a new set of training data to give to each *DecisionTree*, it would be more efficient to use the same data set for all trees, and simply assign random weights.  Third, it would make sense to train multiple *DecisionTree*s in parallel (although I note that sklearn's version performs faster even without parallel computing).

My RandomForest also lacks flexibility, in that it only works for binary labels, and numerical features.
# -*- coding: utf-8 -*-
"""
Created on August 31, 2018

@author: Tristan
"""

import time
import random
import pandas as pd
import numpy as np

class TreeNode():
    """A TreeNode is a single node in a decision tree.
    
    Instance variables:
    is_leaf -- whether this node is a leaf or internal node
    children -- if this is not a leaf, this is a list of the two child nodes
    col -- if this is not a leaf, this is the index of the feature used to make a decision
    threshold -- if this is not a leaf, it is the threshold on the feature used to make a decision
    guess -- if this is a leaf, a single guess for all data arriving at leaf
    address -- if this is a leaf, this is an integer assigned such that all leaves of a tree
        have a unique address from 0 to N-1
    """
    #initiate node as a leaf
    def __init__(self, guess, address):
        """Initialize a TreeNode as a leaf"""
        self.is_leaf = True
        self.guess = guess
        self.address = address
    
    def make_children(self, df, feature_cols, label_col, num_leaves, weight = 0.5):
        """Choose a partition, and produce two children.  Returns True if successful
        
        Parameters:
        df -- the subset of data that arrives at this leaf
        feature_cols -- the slice referring to the feature columns
        label_col -- the index of the label column
        num_leaves -- total number of leaves in the tree so far
        weight -- the weight assigned to correctly guessing labels with value of 1.
            Correctly guessing labels with value of 0 is 
            assigned a weight of (1-weight). (default 0.5)
        """
        
        if not self.is_leaf:
            #Fails because this TreeNode is not a leaf.
            return False
        
        least_errors = np.inf
        best_thresh = 0
        best_above = 0
        best_below = 0
        best_col = 0
        #loop in a random order so that if there are a lot of ties,
        #it doesn't bias it towards early columns
        cols = list(feature_cols)
        random.shuffle(cols)
        for col in cols:
            incorrect, threshold, above, below = greedy_partition(df, col, label_col)
            
            #print("column",col,":",incorrect, threshold, above, below)
            if incorrect < least_errors:
                least_errors = incorrect
                best_thresh = threshold
                best_above = above
                best_below = below
                best_col = col
        
        if np.isinf(least_errors):
            #No partition can be made because all data are identical
            return False
        
        #print(best_col,best_thresh,best_above,best_below,least_errors)
        self.col = best_col
        self.threshold = best_thresh
        self.children = [TreeNode(best_below, self.address), TreeNode(best_above, num_leaves)]
        self.is_leaf = False
        return True
        
    def predict(self, row):
        """Given a single row, returns the prediction
        and the address of the leaf making the prediction"""
        leaf = self.get_leaf(row)
        return leaf.guess, leaf.address
    
    def get_leaf(self, row):
        """Given a single row, returns a pointer to the leaf where it ends up"""
        if self.is_leaf:
            return self
        elif row[self.col] > self.threshold:
            return self.children[1].get_leaf(row)
        else:
            return self.children[0].get_leaf(row)
    
    def summarize(self):
        """A recursive method to print out the structure of the tree"""
        if self.is_leaf:
            return str(self.guess)
        else:
            return "["+self.children[0].summarize()+","+self.children[1].summarize()+"]"

def greedy_partition(df, feature_col, label_col, weight=0.5):
    """Given a subset of the data and a single feature, determines the best partition.
    
    Parameters:
    df -- subset of data under consideration
    feature_col -- index of feature column
    label_col -- index of label column
    weight -- the weight assigned to correctly guessing labels with value of 1.
        Correctly guessing labels with value of 0 is 
        assigned a weight of (1-weight). (default 0.5)
    
    Returns:
    incorrect -- the (weighted) number of incorrect guesses under this partition
    threshold -- the place to put the partition
    above -- a boolean indicating whether guesses above the threshold are 0 or 1
    below -- a boolean indicating whether guesses below the threshold are 0 or 1
    """
    
    #first get the features
    features = df.iloc[:, feature_col]
    size = len(features)
    #offset the labels in order to weight them properly
    labels = df.iloc[:, label_col] + weight - 1
    #now sort them by feature value
    sorter = np.argsort(features)
    features = features.iloc[sorter]
    labels = labels.iloc[sorter]
    
    #next we want to group data points whose features are identical
    unique_features = []
    weighted_labels = []
    previous_feature = np.nan
    for i, feature in enumerate(features):
        if feature == previous_feature:
            weighted_labels[-1] += labels.iloc[i]
        else:
            unique_features.append(feature)
            weighted_labels.append(labels.iloc[i])
            previous_feature = feature
    
    if len(unique_features) == 1:
        #if all features are identical, then no partition can be made
        return np.inf, 0, 0, 0
    
    #next I create a magic list (called "optimizer") which computes a goodness metric
    #for any choice of partition.
    #If the best partition (on unique_features) is between index i and i+1, 
    #then abs(optimizer) is maximized at index i
    optimizer = np.cumsum(weighted_labels)
    weight_sum = optimizer[-1]
    optimizer -= weight_sum/2
    optimizer[-1] = 0 #don't want to make a partition after last element
    
    threshold_index = np.argmax(abs(optimizer))
    optimizer_value = optimizer[threshold_index]
    if abs(optimizer_value) < abs(weight_sum)/2:
        #under this condition, it's best to guess the same thing on both sides of partition
        if weight_sum > 0:
            #guess 1 on both sides
            above = 1
            below = 1
            incorrect = size*weight*(1-weight) + weight_sum*(weight-1)
        else:
            #guess 0 on both sides
            above = 0
            below = 0
            incorrect = size*weight*(1-weight) + weight_sum*weight
        #either way, set threshold to the median
        threshold = (features.iloc[int((size-1)/2)]+features.iloc[int(size/2)])/2
    else:
        incorrect = -abs(optimizer_value) + size*weight*(1-weight) + weight_sum*(weight-0.5)
        threshold = (unique_features[threshold_index] + unique_features[threshold_index+1])/2
        above = 0 if (optimizer_value > 0) else 1
        below = 1-above
    
    return incorrect, threshold, above, below
    
class DecisionTree():
    """Single decision tree
    
    Instance variables:
    root -- pointer to the root TreeNode
    features -- slice referring to the feature columns
    col -- index of column with label
    num_leaves -- current number of leaves in the tree
    addresses -- A list of the address of the leaf used to classify each row
    incorrects -- A list of the number of incorrect guesses for each leaf;
        Leaves that cannot produce children are set to -1
    predictions -- A list of predictions for the labels of each row
    runtime -- current runtime for the training of this tree
    verbose -- if true, prints out information as tree is trained
    weight -- the weight assigned to correctly guessing labels with value of 1.
        Correctly guessing labels with value of 0 is 
        assigned a weight of (1-weight).
    """
    
    def __init__(self, df, feature_cols, label_col, weight=0.5):
        """Initializes a DecisionTree on a given data set"""
        self.root = TreeNode(0, 0)
        self.features = feature_cols
        self.col = label_col
        self.num_leaves = 1
        
        self.addresses = np.zeros(df.shape[0])
        self.incorrects = np.array([df.iloc[:, label_col].sum()])
        self.predictions = np.ones(df.shape[0])
        self.runtime = 0
        self.verbose = False
        self.weight = weight
    
    def update(self, df, node):
        """Updates instance variables addresses, incorrects, and predictions, in light of new leaves
        
        Parameters:
        df -- pointer to the DataFrame containing data
        node -- The former leaf node which has just produced two new children
        """
        
        old_address = node.address
        new_address = node.children[1].address
        left_guess = node.children[0].guess
        right_guess = node.children[1].guess
        
        col_list = list(df.columns)
        changed_rows = [i for i in range(df.shape[0]) if self.addresses[i] == old_address]
        prediction_results = [node.predict(row) for i,row in enumerate(df[col_list].values)
                              if self.addresses[i] == old_address]
        self.predictions[changed_rows] = [result[0] for result in prediction_results]
        self.addresses[changed_rows] = [result[1] for result in prediction_results]
        if left_guess == 1:
            modified_incorrects = sum([(1-label) for result, label in 
                                       zip(prediction_results, df.iloc[changed_rows, self.col])
                                       if result[1] == old_address])*(1-self.weight)
        else:
            modified_incorrects = sum([label for result, label in 
                                       zip(prediction_results, df.iloc[changed_rows, self.col])
                                       if result[1] == old_address])*self.weight
        if right_guess == 1:
            new_incorrects = sum([(1-label) for result, label in 
                                  zip(prediction_results, df.iloc[changed_rows, self.col])
                                  if result[1] == new_address])*(1-self.weight)
        else:
            new_incorrects = sum([label for result, label in 
                                  zip(prediction_results, df.iloc[changed_rows, self.col])
                                  if result[1] == new_address])*self.weight
        
        '''modified_incorrects = 0
        new_incorrects = 0
        for i,row in enumerate(df.shape[0]):
            curr_address = self.addresses[i]
            if self.addresses[i] == address:
                self.predictions[i], self.addresses[i] = node.predict(df.iloc[i, :])
        
        errors = (df.iloc[:, self.col] != self.predictions)
        modified_incorrects = (errors & (self.addresses == node.children[0].address)).sum()
        new_incorrects = (errors & (self.addresses == node.children[1].address)).sum()
        '''
        self.incorrects[old_address] = modified_incorrects
        self.incorrects = np.append(self.incorrects, new_incorrects)
    
    def grow_step(self, df):
        """Finds worst-performing leaf, and has that leaf produce new children.
        """
        
        #if all the guesses are correct, then clearly the algorithm should halt.
        worst = self.incorrects.max()
        
        #end early if there's nothing to be done
        if worst == 0:
            return# 0
        
        worst_i = self.incorrects.argmax()
        #get a pointer to the leaf itself
        row = df.iloc[np.argmax(self.addresses == worst_i), :]
        worst_leaf = self.root.get_leaf(row)
        
        #now, give the leaf new children
        success = worst_leaf.make_children(df.loc[pd.IndexSlice[(self.addresses == worst_i)], :],
                                           self.features, self.col, self.num_leaves, weight=self.weight)
        if success:
            #if children were produced successfully
            self.num_leaves += 1
            #recalculate predictions
            self.update(df, worst_leaf)

            if self.verbose:
                if worst_leaf.children[0].guess == worst_leaf.children[1].guess:
                    print("Leaf %i produces children using column %i (chosen at random)."
                          % (worst_leaf.address, worst_leaf.col))
                else:
                    print("Leaf %i produces children using column %i."
                          % (worst_leaf.address, worst_leaf.col))
                print("Weighted error in this branch: %.1f before; %.1f after"
                      % (worst, self.incorrects[worst_i]+self.incorrects[-1]))
        else:
            #if children were not produced successfully
            #that means that the leaf cannot be improved, and should be put on an ignore list
            self.incorrects[worst_i] = -1
            if self.verbose:
                print("Leaf %i failed to produce children.")
    
    def grow_tree(self, df, num_iter, tolerance):
        """Grows the tree by running grow_step repeatedly
        
        Parameters:
        df -- pointer to the DataFrame containing data
        num_iter -- maximum number of iterations
        tolerance -- iterations halt if the worst leaf is doing better than this tolerance level.
        """
        start_time = time.time()
        for i in range(num_iter):
            if self.incorrects.max() <= tolerance:
                break
            #worst = self.grow_step(df)
            self.grow_step(df)
                
        self.runtime += (time.time() - start_time)/60
    
    def predict_labels(self, eval_data):
        """Predicts the labels for a set of evaluation data"""
        col_list = list(eval_data.columns)
        predictions = np.zeros(eval_data.shape[0])
        predictions[:] = [self.root.predict(row)[0] for row in eval_data[col_list].values]
        #predictions = np.zeros(eval_data.shape[0])
        #for i in range(eval_data.shape[0]):
        #    predictions[i] = self.root.predict(eval_data.iloc[i, :])[0]
        return predictions
        
    def confusion(self, eval_data):
        """Returns a confusion matrix for the predictions on a set of evaluation data.
        Rows correspond to true labels, columns correspond to predictions.
        First row/col is for 1s, the second row/col is for 0s.
        """
        
        c_matrix = np.zeros((2, 2))
        predictions = self.predict_labels(eval_data)
        c_matrix[0, 0] = (predictions*eval_data.iloc[:, self.col]).sum()
        c_matrix[1, 0] = (predictions*(1-eval_data.iloc[:, self.col])).sum()
        c_matrix[0, 1] = ((1-predictions)*eval_data.iloc[:, self.col]).sum()
        c_matrix[1, 1] = ((1-predictions)*(1-eval_data.iloc[:, self.col])).sum()
        temp = c_matrix.sum()
        c_matrix /= temp
        return c_matrix
    
    def evaluate(self, eval_data):
        """calculates the fraction of correct predictions on an evaluation set."""
        return np.trace(self.confusion(eval_data))

#a model that uses a bunch of bagged decision trees
class RandomForest():
    """A model that creates a bunch of bagged decision trees.
    
    Instance variables:
    features -- slice referring to the feature columns
    col -- index of label column
    target_runtime -- the runtime allotted to training this forest
    fix_iter -- If true, then fixes the number of iterations per tree.
        If false, fixes the number of trees.
    num_iter -- if fix_iter is true, this is the max number of iterations per tree
    num_trees -- if fix_iter is false, this is the number of trees to train
    num_features -- randomly assigns each tree this number of features, randomly selected
    runtime -- time spent training so far
    checkpoint -- a checkpoint for the timer
    forest -- a list of pointers to the DecisionTrees
    weight -- the weight assigned to correctly guessing labels with value of 1.
        Correctly guessing labels with value of 0 is 
        assigned a weight of (1-weight).
    """
    
    def __init__(self, feature_cols, label_col, iterations, num_trees,
                 num_features, target_runtime, fix_iter=True, weight=0.5):
        """Initializes RandomForest settings."""
        self.features = feature_cols
        self.col = label_col
        self.num_iter = iterations
        self.num_trees = num_trees
        self.num_features = num_features
        self.target_runtime = target_runtime
        self.fix_iter = fix_iter
        self.weight = weight
        
        self.runtime = 0
        self.checkpoint = 0
        self.forest = []
    
    def start_timer(self):
        """Starts a timer for purpose of measuring runtime"""
        self.checkpoint = time.time()
    
    def update_runtime(self):
        """Updates the current runtime"""
        self.runtime += (time.time() - self.checkpoint)/60
        self.start_timer()
    
    def train(self, df):
        """Trains the forest"""
        #if the forest has already been trained, then we train it by just adding more trees!
        old_runtime = self.runtime
        self.runtime = 0
        self.start_timer()
        
        if not self.fix_iter:
            #case 1: We have a target number of trees,
            #and we choose the number of iterations to match a target runtime
            #Grow the first tree carefully, trying to estimate the appropriate number of iterations
            bagged_df = bag(df)
            
            curr_tree = DecisionTree(bagged_df, self.subspace(), self.col, self.weight)
            self.forest.append(curr_tree)
            curr_tree.grow_tree(bagged_df, 10, 0) #just ten iterations at first
            
            self.update_runtime()
            #estimate the number of iterations that we can still do
            #I'm putting a floor on the estimated runtime per iteration at 0.001 minutes.
            #Just so the trees don't get away with themselves
            self.num_iter = round(10*self.target_runtime / self.num_trees / max(self.runtime, 0.01))
            if self.num_iter > 10:
                curr_tree.grow_tree(bagged_df, self.num_iter-10, 0)
            
            #make a final estimate of the number of iterations
            self.update_runtime()
            self.num_iter = round(self.num_iter*self.target_runtime / self.num_trees
                                  / max(self.runtime, 0.001*self.num_iter))
                
            for i in range(1, self.num_trees):
                bagged_df = bag(df)
                
                curr_tree = DecisionTree(bagged_df, self.subspace(), self.col, self.weight)
                self.forest.append(curr_tree)
                curr_tree.grow_tree(bagged_df, self.num_iter, 0)
                
                self.update_runtime()
            
        else:
            #case 2: We have a fixed number of iterations,
            #and we choose the number of trees to match a target runtime
            #just keep running until it overshoots the target runtime
            while self.runtime < self.target_runtime:
                bagged_df = bag(df)
                
                curr_tree = DecisionTree(bagged_df, self.subspace(), self.col, self.weight)
                #curr_tree = DecisionTree(bagged_df, self.features, self.col)
                self.forest.append(curr_tree)
                curr_tree.grow_tree(bagged_df, self.num_iter, 0)
                
                self.update_runtime()
            
            self.num_trees = len(self.forest)
        
        self.runtime += old_runtime
        print("""Forest planted in %.2f minutes. %i trees grown with %i iterations and %i features."""
              % (self.runtime, self.num_trees, self.num_iter, self.num_features))
    
    def subspace(self):
        """Returns a subspace of the feature space"""
        return np.random.choice(self.features, self.num_features, replace=False)
            
    def predict_labels(self, eval_data):
        """Predicts the labels for a set of evaluation data."""
        col_list = list(eval_data.columns)
        votes = np.zeros(eval_data.shape[0])
        predictions = votes.copy()
        for tree in self.forest:
            votes += [tree.root.predict(row)[0] for row in eval_data[col_list].values]
            
        predictions[:] = [(1 if num_votes > self.num_trees/2 else 0)
                                  for num_votes in votes]
        #predictions = np.zeros(eval_data.shape[0])
        #for i in range(eval_data.shape[0]):
        #    votes = 0
        #    for tree in self.forest:
        #        votes += tree.root.predict(eval_data.iloc[i, :])[0]
        #    predictions[i] = 1 if votes > self.num_trees/2 else 0
            
        return predictions
        
    
    def confusion(self, eval_data):
        """Returns a confusion matrix for the predictions.
        Rows correspond to true labels, columns correspond to predictions.
        First row/col is for 1s, the second row/col is for 0s.
        """
        c_matrix = np.zeros((2, 2))
        predictions = self.predict_labels(eval_data)
        c_matrix[0, 0] = (predictions*eval_data.iloc[:, self.col]).sum()
        c_matrix[1, 0] = (predictions*(1-eval_data.iloc[:, self.col])).sum()
        c_matrix[0, 1] = ((1-predictions)*eval_data.iloc[:, self.col]).sum()
        c_matrix[1, 1] = ((1-predictions)*(1-eval_data.iloc[:, self.col])).sum()
        temp = c_matrix.sum()
        c_matrix /= temp
        return c_matrix
    
    def evaluate(self, eval_data):
        """Calculates the fraction of correct predictions on an evaluation set"""
        return np.trace(self.confusion(eval_data))
    
    def train_valid(self, train_data, valid_data):
        """Trains and evaluates the forest based on a validation set"""
        self.train(train_data)
        print("Accuracy: %.3f" % self.evaluate(valid_data))
            
def bag(df):
    """Returns a subset of the data, chosen with replacement, same size as original set"""
    subset = np.random.choice(range(len(df)), len(df), replace=True)
    return df.iloc[subset, :]
    
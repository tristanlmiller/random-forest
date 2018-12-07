# -*- coding: utf-8 -*-
"""
Created on August 31, 2018

@author: Tristan
"""

import pandas as pd
import numpy as np
import time
import random

#a tree_node is a single node in a decision tree.
#leaves have a guess (a 0 or 1), and an address
#The addresses are chosen such that all the leaves are given a unique address from 0 to N-1.
#Internal nodes contain pointers to two children, and a description of how to decide between the two children.
class tree_node(object):
    #initiate node as a leaf
    def __init__(self,guess,address):
        self.is_leaf = True
        self.guess = guess
        self.address = address
    
    #If the node is a leaf, this function uses a greedy algorithm to produce the best partition, and produces two children.
    def make_children(self,df,feature_cols,label_col,num_leaves):
        #df is assumed to be the subset of data arriving at this node
        #feature_cols is a slice of columns numbers of the features
        #label_col is the column number of the label being predicted
        #num_leaves is the total number of leaves in the tree, before making children
        
        if(not self.is_leaf):
            print("Error: tree_node should not overwrite its children")
        
        least_errors = np.inf
        best_thresh = 0
        best_above = 0
        best_below = 0
        best_col = 0
        #loop in a random order so that if there are a lot of ties, it doesn't bias it towards early columns
        cols = list(feature_cols)
        random.shuffle(cols)
        for col in cols:
            incorrect, threshold, above, below = greedy_partition(df,col,label_col)
            
            #print("column",col,":",incorrect, threshold, above, below)
            if incorrect < least_errors:
                least_errors = incorrect
                best_thresh = threshold
                best_above = above
                best_below = below
                best_col = col
        
        if(np.isinf(least_errors)):
            #I think this can only happen if literally all the features are identical.  Not likely.
            print("Error: no partition can be made")
        
        #print(best_col,best_thresh,best_above,best_below,least_errors)
        self.col = best_col
        self.threshold = best_thresh
        self.children = [tree_node(best_below,self.address),tree_node(best_above,num_leaves)]
        self.is_leaf = False
        
    
    #makes predictions based on this node
    #also returns address of leaf used to make prediction
    def predict(self,row):
        leaf = self.get_leaf(row)
        return leaf.guess, leaf.address
    
    #given a row, returns a pointer to the leaf used to classify the row
    def get_leaf(self,row):
        if self.is_leaf:
            return self
        elif row[self.col] > self.threshold:
            return self.children[1].get_leaf(row)
        else:
            return self.children[0].get_leaf(row)
        
    #A recursive method to print out the structure of the tree
    #will look something like [[0,1],[0,[0,1]]]
    def structure(self):
        if self.is_leaf:
            return str(self.guess)
        else:
            return "["+self.children[0].summarize()+","+self.children[1].summarize()+"]"
        
    
#Given a particular feature and a subset of the data, figures out the best place to make a partition.
#returns 4 parts: 
#incorrect, the number of incorrect guesses under this partition
#threshold, the place to put the partition
#above, a boolean indicating whether guesses above the threshold are 0 or 1,
#and below, a boolean indicating whether guesses below the threshold are 0 or 1.
def greedy_partition(df,feature_col,label_col):
    #I'm sure there must be a standard algorithm to do this
    #but I'm just doing it the obvious way, by sorting the list, and testing each possible partition
    
    features = df.iloc[:,feature_col]
    labels = df.iloc[:,label_col]*2-1
    sorter = np.argsort(features)
    features = features.iloc[sorter]
    labels = labels.iloc[sorter]
    
    #the optimizer is a funcion whose absolute value is larger when there are fewer errors
    #it's positive when the guess below the threshold is 1
    diff = labels.sum()
    optimizer = labels.cumsum() - diff/2
    
    #there are frequently duplicate values in the features list
    #I only want to consider partitions between rows that have distinct feature values
    unique_i = np.unique(list(features), return_index=True)[1]
    if(len(unique_i)==1):
        #All features are identical in this branch, so there's no point in using it
        return np.inf, 0, 0, 0
    else:
        allowable_opt = optimizer.iloc[unique_i[1:]-1]
        threshold_i = unique_i[np.argmax(list(abs(allowable_opt)))+1]-1
        
        if(abs(optimizer.iloc[threshold_i]) < abs(diff)/2):
            #This is the condition where it's better to guess the same thing on both sides of the partition
            #The partition doesn't matter, so just suggest the median
            incorrect = (-abs(diff) + len(labels))/2
            threshold = features.median()
            above = 1 if (diff > 0) else 0
            below = above
        else:
            incorrect = -abs(optimizer.iloc[threshold_i]) + len(labels)/2
            threshold = (features.iloc[threshold_i]+features.iloc[threshold_i+1])/2
            above = 0 if (optimizer.iloc[threshold_i] > 0) else 1
            below = 1-above
    
    return incorrect, threshold, above, below
    
    
#This is a single decision tree.  Contains a pointer to the root, and other info needed for the training process
class decision_tree(object):
    def __init__(self,df,feature_cols,label_col):
        self.root = tree_node(0,0)
        self.features = feature_cols
        self.col = label_col
        self.num_leaves = 1
        
        self.addresses = np.zeros(df.shape[0]) #a list of the addresses of leaves used for each data entry
        self.incorrects = np.array([df.iloc[:,label_col].sum()]) #a list of the number of incorrect guesses for each leaf
        self.predictions = np.ones(df.shape[0])
        self.runtime = 0
        self.verbose = False
    
    #Updates predictions, addresses, and incorrects, in light of a leaf making two new children.
    #requires a pointer to the appropriate node
    def update(self,df,node):
        address = node.address
        for i in range(df.shape[0]):
            if self.addresses[i] == address:
                self.predictions[i],self.addresses[i] = node.predict(df.iloc[i,:])
        
        errors = (df.iloc[:,self.col] != self.predictions)
        self.incorrects[address] = (errors & (self.addresses == node.children[0].address)).sum()
        self.incorrects = np.append(self.incorrects, (errors & (self.addresses == node.children[1].address)).sum() )
    
    #finds the worst-performing leaf (as measured by the number of incorrect guesses),
    #and tells that leaf to produce new children.
    #returns number of incorrect guesses of the worst leaf *before* it produced children.
    def grow_step(self,df):
        #if all the guesses are correct, then clearly the algorithm should halt.
        worst = self.incorrects.max()
        
        #end early if there's nothing to be done
        if worst == 0:
            return 0
        
        worst_i = self.incorrects.argmax()
        #get a pointer to the leaf itself
        row = df.iloc[np.argmax(self.addresses == worst_i),:]
        worst_leaf = self.root.get_leaf(row)
        
        #now, give the leaf new children
        worst_leaf.make_children( df.loc[pd.IndexSlice[(self.addresses == worst_i)],:],self.features,self.col,self.num_leaves )
        self.num_leaves += 1
        
        #recalculate predictions
        self.update(df,worst_leaf)
        
        if(self.verbose):
            if(worst_leaf.children[0].guess == worst_leaf.children[1].guess):
                print("Leaf %i produces children using column %i (chosen at random)." % (worst_leaf.address, worst_leaf.col))
            else:
                print("Leaf %i produces children using column %i." % (worst_leaf.address, worst_leaf.col))
            print("incorrect guesses in this branch: %i before; %i after" % (worst, self.incorrects[worst_i]+self.incorrects[-1]))
        
        return worst
    
    #Grows the tree by iteratively finding the worst leaf and telling it to produce children
    #stops after num_iter, or when the worst leaf is better than the tolerance level, whichever comes first
    def grow_tree(self,df,num_iter,tolerance):
        start_time = time.time()
        for i in range(num_iter):
            if self.incorrects.max() <= tolerance:
                break
            worst = self.grow_step(df)
                
        self.runtime += (time.time() - start_time)/60
    
    #predicts the labels for a set of evaluation data
    def predict_labels(self,eval_data):
        predictions = np.zeros(eval_data.shape[0])
        for i in range(eval_data.shape[0]):
            predictions[i] = self.root.predict(eval_data.iloc[i,:])[0]
        return predictions
        
    
    #returns a confusion matrix for the predictions
    #rows correspond to true labels, columns correspond to predictions
    #first row/col is for 1s, the second row/col is for 0s.
    def confusion(self,eval_data):
        c_matrix = np.zeros((2,2))
        predictions = self.predict_labels(eval_data)
        c_matrix[0,0] = (predictions*eval_data.iloc[:,self.col]).sum()
        c_matrix[1,0] = (predictions*(1-eval_data.iloc[:,self.col])).sum()
        c_matrix[0,1] = ((1-predictions)*eval_data.iloc[:,self.col]).sum()
        c_matrix[1,1] = ((1-predictions)*(1-eval_data.iloc[:,self.col])).sum()
        temp = c_matrix.sum()
        c_matrix /= temp
        return c_matrix
    
    #calculates the fraction of correct predictions on an evaluation set
    def evaluate(self,eval_data):
        return np.trace(self.confusion(eval_data))

#a model that uses a bunch of bagged decision trees
class random_forest(object):
    def __init__(self,feature_cols,label_col,iterations,num_trees,num_features,target_runtime,fix_iter=True):
        self.features = feature_cols
        self.col = label_col
        self.num_iter = iterations
        self.num_trees = num_trees
        self.num_features = num_features
        self.target_runtime = target_runtime
        self.fix_iter = fix_iter
        
        self.runtime = 0
        self.checkpoint = 0
        self.forest = []
    
    #starts a timer for purpose of measuring runtime
    def start_timer(self):
        self.checkpoint = time.time()
    
    #updates the current runtime
    def update_runtime(self):
        self.runtime += (time.time() - self.checkpoint)/60
        self.start_timer()
    
    #trains the forest
    def train(self,df):
        #if the forest has already been trained, then we train it by just adding more trees!
        old_runtime = self.runtime
        self.runtime = 0
        self.start_timer()
        
        if not self.fix_iter:
            #case 1: We have a target number of trees, and we choose the number of iterations to match a target runtime
            #Grow the first tree carefully, trying to estimate the appropriate number of iterations
            bagged_df = self.bag(df)
            subspace = self.subspace()
            
            curr_tree = decision_tree(bagged_df,self.features,self.col)
            self.forest.append(curr_tree)
            curr_tree.grow_tree(bagged_df,10,0) #just ten iterations at first
            
            self.update_runtime()
            #estimate the number of iterations that we can still do
            #I'm putting a floor on the estimated runtime per iteration at 0.001 minutes.
            #Just so the trees don't get away with themselves
            self.num_iter = round(10*self.target_runtime / self.num_trees / max(self.runtime,0.01))
            if(self.num_iter > 10):
                curr_tree.grow_tree(bagged_df,num_iter-10,0)
            
            #make a final estimate of the number of iterations
            self.update_runtime()
            self.num_iter = round(self.num_iter*self.target_runtime / self.num_trees / max(self.runtime,0.001*self.num_iter))
                
            for i in range(1,self.num_trees):
                bagged_df = self.bag(df)
                subspace = self.subspace()
                
                curr_tree = decision_tree(bagged_df,self.features,self.col)
                self.forest.append(curr_tree)
                curr_tree.grow_tree(bagged_df,self.num_iter,0)
                
                self.update_runtime()
            
        else:
            #case 2: We have a fixed number of iterations, and we choose the number of trees to match a target runtime
            #just keep running until it overshoots the target runtime
            while self.runtime < self.target_runtime:
                bagged_df = self.bag(df)
                subspace = self.subspace()
                
                curr_tree = decision_tree(bagged_df,self.features,self.col)
                self.forest.append(curr_tree)
                curr_tree.grow_tree(bagged_df,self.num_iter,0)
                
                self.update_runtime()
            
            self.num_trees = len(self.forest)
        
        self.runtime += old_runtime
        print("Forest planted in %.2f minutes.  %i trees grown with %i iterations and %i features." %
              (self.runtime, self.num_trees, self.num_iter, self.num_features))
     
    #returns a subset of the data, chosen with replacement, same size as original set
    def bag(self,df):
        subset = np.random.choice(range(len(df)),len(df),replace=True)
        return df.iloc[subset,:]
    
    #returns a subspace of the feature space
    def subspace(self):
        return np.random.choice(self.features,self.num_features,replace=False)
            
    #predicts the labels for a set of evaluation data
    def predict_labels(self,eval_data):
        predictions = np.zeros(eval_data.shape[0])
        for i in range(eval_data.shape[0]):
            votes = 0
            for tree in self.forest:
                votes += tree.root.predict(eval_data.iloc[i,:])[0]
            predictions[i] = 1 if votes > self.num_trees/2 else 0
            
        return predictions
        
    
    #returns a confusion matrix for the predictions
    #rows correspond to true labels, columns correspond to predictions
    def confusion(self,eval_data):
        c_matrix = np.zeros((2,2))
        predictions = self.predict_labels(eval_data)
        c_matrix[0,0] = (predictions*eval_data.iloc[:,self.col]).sum()
        c_matrix[1,0] = (predictions*(1-eval_data.iloc[:,self.col])).sum()
        c_matrix[0,1] = ((1-predictions)*eval_data.iloc[:,self.col]).sum()
        c_matrix[1,1] = ((1-predictions)*(1-eval_data.iloc[:,self.col])).sum()
        temp = c_matrix.sum()
        c_matrix /= temp
        return c_matrix
    
    #calculates the fraction of correct predictions on an evaluation set
    def evaluate(self,eval_data):
        return np.trace(self.confusion(eval_data))
    
    #trains and evaluates the forest based on a validation set
    def train_valid(self,train_data,valid_data):
        self.train(train_data)
        print("Accuracy: %.3f" % self.evaluate(valid_data))
            
            
            
            
            
            
            
            

import numpy as np
import pandas as pd

from numpy import log2 as log

class ID3Classifier:

  def __init__(self, max_depth=10, gain=0, missing_value=False):
    self.max_depth = max_depth
    self.gain = gain
    self.missing_value = missing_value
    self.tree = None


  def fit(self, input, output):
    data = input.copy()
    data[output.name] = output
    self.tree = self.decision_tree(data, data, input.columns, output.name)

  def predict(self, input):
    # convert input data into a dictionary of samples
    samples = input.to_dict(orient='records')
    predictions = []

    # make a prediction for every sample
    for sample in samples:
      predictions.append(self.make_prediction(sample, self.tree, 1.0))

    return predictions

  def make_prediction(self, sample, tree, default=1): 
        # map sample data to tree
        for attribute in list(sample.keys()):
        # check if feature exists in tree
            if attribute in list(tree.keys()):
                try:
                    result = tree[attribute][sample[attribute]]
                except:
                    return default
                result = tree[attribute][sample[attribute]]
        # if more attributes exist within result, recursively find best result
        if isinstance(result, dict):
          return self.make_prediction(sample, result)
        else:
          return result

  def entropy(self, attribute_column):
    # find unique values and their frequency counts for the given attribute
    values, counts = np.unique(attribute_column, return_counts=True)

    # calculate entropy for each unique value
    entropy_list = []

    for i in range(len(values)):
      probability = counts[i]/np.sum(counts)
      entropy_list.append(-probability*np.log2(probability))

    # calculate sum of individual entropy values
    total_entropy = np.sum(entropy_list)

    return total_entropy

  def information_gain(self, data, feature_attribute_name, target_attribute_name):
    # find total entropy of given subset
    total_entropy = self.entropy(data[target_attribute_name])

    # find unique values and their frequency counts for the attribute to be split
    values, counts = np.unique(data[feature_attribute_name], return_counts=True)

    # calculate weighted entropy of subset
    weighted_entropy_list = []

    for i in range(len(values)):
      subset_probability = counts[i]/np.sum(counts)
      subset_entropy = self.entropy(data.where(data[feature_attribute_name]==values[i]).dropna()[target_attribute_name])
      weighted_entropy_list.append(subset_probability*subset_entropy)

    total_weighted_entropy = np.sum(weighted_entropy_list)

    # calculate information gain
    information_gain = total_entropy - total_weighted_entropy

    return information_gain

  def majority_error(self, attribute_column):
    # find unique values and their frequency counts for the given attribute
    values, counts = np.unique(attribute_column, return_counts=True)
    #find index of the error
    index = np.where((values == "unacc") | (values == "no") )
    #save value 
    error_value = counts[index]
    # calculate entropy for each unique value
    majority_list = []

    for i in range(len(values)):
      probability = counts[i]/np.sum(counts)
      majority_err = error_value/np.sum(counts)
      majority_list.append(probability*majority_err)

    # calculate sum of individual entropy values
    total_majority_error = np.sum(majority_list)

    return total_majority_error


  def majority_gain(self, data, feature_attribute_name, target_attribute_name):
    
    # find total majority gain of given subset
    total_majority = self.majority_error(data[target_attribute_name])

    # find unique values and their frequency counts for the attribute to be split
    values, counts = np.unique(data[feature_attribute_name], return_counts=True)

    majority_error_list = []

    for i in range(len(values)):
      subset_probability = counts[i]/np.sum(counts)
      majority_error_list.append(subset_probability*self.majority_error(data.where(data[feature_attribute_name]==values[i]).dropna()[target_attribute_name]))

    majority_error = np.sum(majority_error_list)

    majority_gain = total_majority - majority_error

    return majority_gain

  def gini_error(self, attribute_column):

    # find unique values and their frequency counts for the attribute to be split
    values, counts = np.unique(attribute_column, return_counts=True)

    gini_error_list = []

    for i in range(len(values)):
      gini_index = (counts[i]/np.sum(counts))**2
      gini_error_list.append(gini_index)
      

    gini_error = np.sum(gini_error_list)

    return gini_error

  def gini_gain(self, data, feature_attribute_name, target_attribute_name):

    # find total majority gain of given subset
    total_gini_error = self.gini_error(data[target_attribute_name])

    # find unique values and their frequency counts for the attribute to be split
    values, counts = np.unique(data[feature_attribute_name], return_counts=True)

    gini_error_list = []

    for i in range(len(values)):
      subset_probability = counts[i]/np.sum(counts)
      gini_error_list.append(subset_probability*self.gini_error(data.where(data[feature_attribute_name]==values[i]).dropna()[target_attribute_name]))

    gini_error = np.sum(gini_error_list)
      
    gini_gain = total_gini_error - gini_error

    return gini_gain


  def decision_tree(self, data, original_data, feature_attribute_names, target_attribute_name, parent_node_class=None, depth=0):
    # base cases:

    # if data is pure, return the majority class of subset
    unique_classes = np.unique(data[target_attribute_name])

    if len(unique_classes) <= 1:
      return unique_classes[0]
      # if subset is empty, ie. no samples, return majority class of original data
    elif len(data) == 0:
      majority_class_index = np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])
      return np.unique(original_data[target_attribute_name])[majority_class_index]
    # if data set contains no features to train with, return parent node class
    elif len(feature_attribute_names) == 0:
      return parent_node_class
    # if none of the above are true, construct a branch:
    else:
    # determine parent node class of current branch
      majority_class_index = np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])
      parent_node_class = unique_classes[majority_class_index]


    # Let us consider "unknown" as attribute value missing. 
    # Here we simply complete it with the majority of other values of the same attribute in the training set.
    if self.missing_value == True:
      # find unique values and their frequency counts for the attribute to be split
      majority_class_index = np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])
      majority_value = np.unique(original_data[feature_attribute_names])[majority_class_index]
      # set where data is "unknown" to the majority value
      data[feature_attribute_names] = np.where(data[feature_attribute_names] == "unknown", majority_value, data[feature_attribute_names])



    # determine information gain values for each feature
    # choose feature which best splits the data, ie. highest value
    if self.gain == 0:
      ig_values = [self.information_gain(data, feature, target_attribute_name) for feature in feature_attribute_names]
      best_feature_index = np.argmax(ig_values)
      best_feature = feature_attribute_names[best_feature_index]
    elif self.gain == 1:
      me_values = [self.majority_gain(data, feature, target_attribute_name) for feature in feature_attribute_names]
      best_feature_index = np.argmax(me_values)
      best_feature = feature_attribute_names[best_feature_index]
    elif self.gain == 2:
      gi_values = [self.gini_gain(data, feature, target_attribute_name) for feature in feature_attribute_names]
      best_feature_index = np.argmax(gi_values)
      best_feature = feature_attribute_names[best_feature_index]


    # create tree structure, empty at first
    tree = {best_feature: {}}

    # remove best feature from available features, it will become the parent node
    feature_attribute_names = [i for i in feature_attribute_names if i != best_feature]

    # if max_depth is achived, return the tree
    if depth >= self.max_depth:
      return tree

    # create nodes under parent node
    parent_attribute_values = np.unique(data[best_feature])
    for value in parent_attribute_values:
      sub_data = data.where(data[best_feature] == value).dropna()

      # call the algorithm recursively
      subtree = self.decision_tree(sub_data, original_data, feature_attribute_names, target_attribute_name, parent_node_class, depth + 1)

      # add subtree to original tree
      tree[best_feature][value] = subtree

    return tree
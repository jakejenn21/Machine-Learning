
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder
from numpy import log2 as log

class ID3Classifier:

  def __init__(self, criterion="gini", max_depth=10, missing_value=False, sample_weights=[], numeric_conv=False, enable_categorical=True):
    self.criterion = criterion
    self.max_depth = max_depth
    self.missing_value = missing_value
    self.sample_weights = sample_weights
    self.numeric_conv = numeric_conv
    self.enable_categorical = enable_categorical
    self.tree = None


  def fit(self, input, output, sample_weights=[]):
    data = input.copy()

    # Let us consider "unknown" as attribute value missing. 
    # Here we simply complete it with the majority of other values of the same attribute in the training set.
    if self.missing_value == True:
      for col in data.columns:
        majority_class_index = np.argmax(np.unique(data[col], return_counts=True)[1])
        majority_value = np.unique(data[col])[majority_class_index]
        if majority_value == "unknown":
          newdata = data[data[col] != "unknown"]
          majority_class_index = np.argmax(np.unique(newdata[col], return_counts=True)[1])
          majority_value = np.unique(newdata[col])[majority_class_index]

        data[col] = np.where(data[col] == "unknown", majority_value, data[col])

    if self.numeric_conv == True:
      numdf = data._get_numeric_data()
      for col in numdf.columns:
        # find unique values and their frequency counts for the attribute to be split
        median = np.median(data[col])
        data[col] = np.where(data[col] >= median, "yes", "no")

    if self.enable_categorical == False:
      cols = data.columns
      lblenc = LabelEncoder()
      for col in cols:
        if not(is_numeric_dtype(data[col])): 
          data[col].astype('category')
          data[col] = lblenc.fit_transform(data[col])
    
    data[output.name] = output
    self.tree = self.decision_tree(data, data, input.columns, output.name)

  def predict(self, input):
    # convert input data into a dictionary of samples
    samples = input.to_dict(orient='records')
    predictions = []

    # make a prediction for every sample
    for sample in samples:
      predictions.append(self.make_prediction(sample, self.tree, "yes"))

    return predictions

  def make_prediction(self, sample, tree, default="yes"): 
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
      entropy_list.append(probability*np.log2(probability))

    # calculate sum of individual entropy values
    total_entropy = -1 * np.sum(entropy_list)

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


  def majority_gain(self, data, feature_attribute_name, target_attribute_name):
    # find unique values and their frequency counts for the attribute to be split
    values, counts= np.unique(data[feature_attribute_name], return_counts=True)

    majority_error_list = []

    for i in range(len(values)):
      # for each values
      probability = counts[i]/np.sum(counts)
      # calculate the me for each of the feature attributes
      labels = data.where(data[feature_attribute_name]==values[i]).dropna()[target_attribute_name]
      errvalues, lblcounts = np.unique(labels, return_counts=True)

      index = np.where((errvalues == "no"))
      errcounts = lblcounts[index]

      subset_probability = errcounts/np.sum(lblcounts)

      majority_error_list.append(subset_probability)


    if len(majority_error_list) == 0:
      majority_gain = 1 - np.max(majority_error_list)
    else:
      majority_gain = 1

    return majority_gain

  def gini_error(self, attribute_column):

    # find unique values and their frequency counts for the attribute to be split
    values, counts = np.unique(attribute_column, return_counts=True)

    gini_error_list = []

    for i in range(len(values)):
      gini_index = (counts[i]/np.sum(counts))**2
      gini_error_list.append(-1*gini_index)
      

    gini_error = np.sum(gini_error_list)

    return 1 - gini_error

  def gini_gain(self, data, feature_attribute_name, target_attribute_name):

    # find total majority gain of given subset
    total_gini_error = self.gini_error(data[target_attribute_name])

    # find unique values and their frequency counts for the attribute to be split
    values, counts = np.unique(data[feature_attribute_name], return_counts=True)

    gini_error_list = []

    for i in range(len(values)):
      subset_probability = counts[i]/np.sum(counts)
      gini_error_list.append(subset_probability*self.gini_error(data.where(data[feature_attribute_name]==values[i]).dropna()[target_attribute_name]))

    gini_error = 1 - sum(gini_error_list)
      
    gini_gain = total_gini_error - gini_error

    return gini_gain


  def decision_tree(self, data, original_data, feature_attribute_names, target_attribute_name, parent_node_class=None, depth = 0):
    # base cases:

    # if data is pure, return the majority class of subset
    unique_classes = np.unique(data[target_attribute_name])

    if depth == self.max_depth:
        return parent_node_class

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


    # ig_values = np.array(ig_values, dtype=float)
    #   if(ig_values.size == 0):
    #     return parent_node_class
    #   else:

    # determine information gain values for each feature
    # choose feature which best splits the data, ie. highest value
    if self.criterion == "ig":
      ig_values = [self.information_gain(data, feature, target_attribute_name) for feature in feature_attribute_names]
      best_feature_index = np.argmax(ig_values)
      best_feature = feature_attribute_names[best_feature_index]

    elif self.criterion == "me":
      me_values = [self.majority_gain(data, feature, target_attribute_name) for feature in feature_attribute_names]
      best_feature_index = np.argmax(me_values)
      best_feature = feature_attribute_names[best_feature_index]

    elif self.criterion == "gini":
      gi_values = [self.gini_gain(data, feature, target_attribute_name) for feature in feature_attribute_names]
      best_feature_index = np.argmax(gi_values)
      best_feature = feature_attribute_names[best_feature_index]


    # create tree structure, empty at first
    tree = {best_feature: {}}

    # remove best feature from available features, it will become the parent node
    feature_attribute_names = [i for i in feature_attribute_names if i != best_feature]

    # create nodes under parent node
    parent_attribute_values = np.unique(data[best_feature])
    for value in parent_attribute_values:
      sub_data = data.where(data[best_feature] == value).dropna()

      # call the algorithm recursively
      subtree = self.decision_tree(sub_data, original_data, feature_attribute_names, target_attribute_name, parent_node_class, depth + 1)
      # add subtree to original tree 
      tree[best_feature][value] = subtree

    return tree
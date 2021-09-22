

import csv
import pprint
import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log

def find_entropy(df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    print(Class)
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy
  
  
def find_entropy_attribute(df,attribute):
  Class = df.keys()[-1]   #To make the code generic, changing target variable class name
  target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
  entropy2 = 0
  for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
  return abs(entropy2)

  def find_majority_error(df):
      #TODO
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    majority_error = 0
    values = df[Class].unique()
    for value in values:
        value = 0
    return entropy

  def find_majority_error_attribute(df,attribute):
      #TODO
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
    variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    entropy2 = 0
    for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
  return abs(entropy2)

  def find_gini_index(df):
      #TODO
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    print(Class)
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy

  def find_gini_index_attribute(df,attribute):
      #TODO
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
    variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
    entropy2 = 0
    for variable in variables:
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
  return abs(entropy2)


def find_IG_winner(df):
    IG = []
    for key in df.keys()[:-1]:
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]

def find_ME_winner(df):
    #TODO
    ME = []
    for key in df.keys()[:-1]:
        ME.append(find_majority_error(df)-find_majority_error_attribute(df,key))
    return df.keys()[:-1][np.argmax(ME)]

def find_GI_winner(df):
    #TODO
    GI = []
    for key in df.keys()[:-1]:
        GE.append(find_gini_index(df)-find_gini_index_attribute(df,key))
    return df.keys()[:-1][np.argmax(GE)]
  
  
def get_subtable(df, node,value):
  return df[df[node] == value].reset_index(drop=True)


def traverse_tree(df, test):
  for key, value in df.items():
        #TODO
        if type(value) is dict:
            traverse_tree(value)
        else:
            print(key, ":", value)


def buildTree(df, depth, gain, tree=None): 
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name

    #if depth hits threashold return tree
    if depth == 0:
        return tree
    
    #Here we build our decision tree

    if gain == 0:
        #Get attribute with maximum information gain
        node = find_IG_winner(df)
    if gain == 1:
        #Get attribute with majority error
        node = find_ME_winner(df)
    if gain == 2:
        #Get attribute with maximum gini Index
        node = find_GI_winnder(df)
    
    #Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])
    
    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
   #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 

    for value in attValue:
        
        subtable = get_subtable(df,node,value)
   
        clValue,counts = np.unique(subtable['y'],return_counts=True)                        
        
        if len(counts)==1:#Checking purity of subset
            tree[node][value] = clValue[0]                                                    
        else:        
            tree[node][value] = buildTree(subtable) #Calling the function recursively 
            depth -= 1
                   
    return tree


with open('DecisionTree/bank/train.csv', 'r') as f:
    age = []
    job = []
    marital = []
    education = []
    default = []
    balance = []
    housing = []
    loan = []
    contact = []
    day = []
    month = []
    duration = []
    campaign = []
    pdays = []
    previous = []
    poutcome = []
    y = []

    for line in f:
        items = line.strip().split(',')

        age.append(items[0])
        job.append(items[1])
        marital.append(items[2])
        education.append(items[3])
        default.append(items[4])
        balance.append(items[5])
        housing.append(items[6])
        loan.append(items[7])
        contact.append(items[8])
        day.append(items[9])
        month.append(items[10])
        duration.append(items[11])
        campaign.append(items[12])
        pdays.append(items[13])
        previous.append(items[14])
        poutcome.append(items[15])
        y.append(items[16])

    dataset ={'age':age,'job':job,'marital':marital,'education':education,'default':default,'balance':balance,'housing':housing,'loan':loan,'contact':contact,'day':day,'month':month,'duration':duration,'campaign':campaign,'pdays':pdays,'previous':previous,'poutcome':poutcome,'y':y}
    df = pd.DataFrame(dataset, columns = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome','y'])
    print(df)


    #calculate information gain trees
    #depth 6
    IGtree6 = buildTree(df, 6, 0)
    #depth 5
    IGtree5 = buildTree(df, 5, 0)
    #depth 4
    IGtree4 = buildTree(df, 4, 0)
    #depth 3
    IGtree3 = buildTree(df, 3, 0)
    #depth 2
    IGtree2 = buildTree(df, 2, 0)
    #depth 1
    IGtree1 = buildTree(df, 1, 0)

    #test
    #TODO

    #calculate majority error trees
    #depth 6
    MEtree6 = buildTree(df, 6, 1)
    #depth 5
    MEtree5 = buildTree(df, 5, 1)
    #depth 4
    MEtree4 = buildTree(df, 4, 1)
    #depth 3
    MEtree3 = buildTree(df, 3, 1)
    #depth 2
    MEtree2 = buildTree(df, 2, 1)
    #depth 1
    MEtree1 = buildTree(df, 1, 1)

    #test
    #TODO


    #calculate gini index trees
    #depth 6
    GItree6 = buildTree(df, 6, 2)
    #depth 5
    GItree5 = buildTree(df, 5, 2)
    #depth 4
    GItree4 = buildTree(df, 4, 2)
    #depth 3
    GItree3 = buildTree(df, 3, 2)
    #depth 2
    GItree2 = buildTree(df, 2, 2)
    #depth 1
    GItree1 = buildTree(df, 1, 2)

    #test
    #TODO

        
  


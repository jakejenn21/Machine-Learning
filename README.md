This is a machine learning library developed by Jacob Jenn for
CS5350/6350 in University of Utah

--------------------------Decision Tree-----------------------

ID3Car.py - Decision Tree test for car dataset (EXECUTE THIS FILE FOR ANALYSIS ON CAR DATASET)

ID3Bank.py - Decision Tree test for car dataset (EXECUTE THIS FILE FOR ANALYSIS ON CAR DATASET)

ID3Classifier.py - Decision tree implementation





Use:

import ID3Classifier as classifier

---initialize model---
model = classifier.ID3Classifier(criterion="gini", max_depth=10, missing_value=True, sample_weights=[], numeric_conv=True, enable_categorical=True)

criterion:

Gain calculation specification when partitioning dataset 

Options:

"ig"- information gain
"me"- majority error
"gini"- gini index

max_depth:

Max depth of the decision tree, default is 10

missing_value:

Treats "unknown" data instances as instead the majority of the feature's attributes

sample_weights:

numeric_conv:

Converts numerical columns to a binary representation

enable_categorical:

True: Enables categorical values for attributes

False: Encodes categorical values for a numberical representation of the dataset

---build decision tree---
model.fit(X_train, y_train)


---predict---
y_pred = model.predict(X_test)





--------------------------Ensemble Learning-----------------------

AdaBoost.py

BankBoost.py



Bag.py

BankBag.py




RandomForest.py

BankForest.py


-------------------------- Linear Regression -----------------------

BatchGradientDescent.py

StochasticGradientDescent.py




ConcreteTests.py






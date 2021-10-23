This is a machine learning library developed by Jacob Jenn for
CS5350/6350 in University of Utah




--------------------------Decision Tree-----------------------

ID3Car.py - Decision Tree test for car dataset (EXECUTE THIS FILE FOR ANALYSIS ON CAR DATASET)

ID3Bank.py - Decision Tree test for car dataset (EXECUTE THIS FILE FOR ANALYSIS ON BANK DATASET)

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

AdaBoost.py - AdaBoost Implementation

Use:

  clf = boost.AdaBoost().fit(X_train, y_train, X_test, y_test, iters)

  pred_train = clf.predict(X_train)


BankBoost.py (EXECUTE THIS FILE FOR ADABOOST ANALYSIS ON BANK DATASET)



Bag.py - Baggin Implementation

Use:

    bagger = Bag.Bag()

    bagger.fit(X_train, y_train, X_test, y_test, B = i, max_depth = 100, min_size = 5, seed = 123)

    y_train_hat = bagger.predict(X_train)

BankBag.py (EXECUTE THIS FILE FOR BAGGING ANALYSIS ON BANK DATASET)




RandomForest.py - Random Forest Implementation

Use:

    model = RandomForest.RandomForest()

    model.fit(X_train, y_train)

    pred_test = model.predict(X_train)



BankForest.py (EXECUTE THIS FILE FOR RANDOM FOREST ANALYSIS ON BANK DATASET)


-------------------------- Linear Regression -----------------------

BatchGradientDescent.py - Batch Gradient Descent Implementation

Use:

    updates, costs, w, learn_rate = sgd.sgd(X_train, y_train, learn_rate=0.0001, n_iter=1000, tolerance=1e-06, batch_size=1)

StochasticGradientDescent.py - Stochastic Gradient Descent Implentation

Use:

   iters, costs, w, learn_rate = bgd.bgd(X_train, y_train, learn_rate=0.001, n_iter=1000, tolerance=1e-06, batch_size=10)


ConcreteTests.py (EXECUTE THIS FILE FOR STOCHASTIC/BATCH GRADIENT DESCENT ANALYSIS ON CONCRETE DATASET)






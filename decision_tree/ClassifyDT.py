from sklearn.naive_bayes import GaussianNB

def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    
    ### your code goes here!
    from sklearn.tree import DecisionTreeClassifier
    clf_2 = DecisionTreeClassifier(min_samples_split=2)
    clf_50 = DecisionTreeClassifier(min_samples_split=50)
    clf_2.fit(features_train, labels_train)
    clf_50.fit(features_train, labels_train)
    return clf
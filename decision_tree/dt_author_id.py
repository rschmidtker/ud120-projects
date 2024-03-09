#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append(".")
from tools.email_preprocess import preprocess
from choose_your_own.class_vis import prettyPicture, output_image
# sys.path.append("../tools/")


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
# features_train, features_test, labels_train, labels_test = preprocess()
features_train, features_test, labels_train, labels_test = preprocess(words_file = "tools/word_data.pkl", authors_file="tools/email_authors.pkl")


print(features_test.shape)

#########################################################
### your code goes here ###
from sklearn.tree import DecisionTreeClassifier
# clf_2 = DecisionTreeClassifier(min_samples_split=2)
clf_x0 = DecisionTreeClassifier(min_samples_split=40)
# clf_2.fit(features_train, labels_train)
clf_x0.fit(features_train, labels_train)
pred = clf_x0.predict(features_test)


from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print(acc)

#########################################################



#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn.naive_bayes import GaussianNB


# import tools.email_preprocess as ep

sys.path.append(".")
from tools.email_preprocess import preprocess
# sys.path.append("../tools/")
# from tools.email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess(words_file = "tools/word_data.pkl", authors_file="tools/email_authors.pkl")


##############################################################
# Enter Your Code Here

### create classifier
clf = GaussianNB()
# clf.fit(labels_train,features_train)

t0 = time()
clf.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")
### use the trained classifier to predict labels for the test features


t0 = time()
pred = clf.predict(features_test)
print ("predict time:", round(time()-t0, 3), "s")


### calculate and return the accuracy on the test data
### this is slightly different than the example, 
### where we just print the accuracy
### you might need to import an sklearn module
accuracy = clf.score(features_test, labels_test)
print( accuracy)
##############################################################

##############################################################
'''
You Will be Required to record time for Training and Predicting 
The Code Given on Udacity Website is in Python-2
The Following Code is Python-3 version of the same code
'''

# t0 = time()
# # < your clf.fit() line of code >
# print("Training Time:", round(time()-t0, 3), "s")

# t0 = time()
# # < your clf.predict() line of code >
# print("Predicting Time:", round(time()-t0, 3), "s")

##############################################################
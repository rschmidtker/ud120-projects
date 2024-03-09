#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append(".")
from tools.email_preprocess import preprocess
from choose_your_own.class_vis import prettyPicture, output_image
# sys.path.append("../tools/")
# from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
# features_train, features_test, labels_train, labels_test = preprocess()
features_train, features_test, labels_train, labels_test = preprocess(words_file = "tools/word_data.pkl", authors_file="tools/email_authors.pkl")


#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(kernel="rbf",C=10000)

#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)


#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''
cnt = 0
print(pred[10], pred[26],pred[50])
for i in pred:
    if i == 1:
        cnt+=1
print(cnt)

#########################################################
from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

# prettyPicture(clf, features_test, labels_test)
# output_image("test.png", "png", open("test.png", "rb").read())

print (acc)

def submitAccuracy():
    return acc
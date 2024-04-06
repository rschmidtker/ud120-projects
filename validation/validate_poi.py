#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
import os
os.chdir(os.path.basename(os.path.dirname(__file__)))
import joblib
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)



### it's all yours from here forward!
import numpy as np  
np.random.seed(42)
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
# clf_2 = DecisionTreeClassifier(min_samples_split=2)
# clf_50 = DecisionTreeClassifier(min_samples_split=50)
clf.fit(features_train, labels_train)
# feat_importance = clf.tree_.compute_feature_importances()
# feat_importance = clf.tree_.compute_feature_importances(normalize=False)
# for i in range(len(feat_importance)):
#     if feat_importance[i] > 0 : print(i, feat_importance[i], fnames[i])
labels_pred = clf.predict(features_test)
# clf_2.fit(features_train, labels_train)
# clf_50.fit(features_train, labels_train)

from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_pred, labels_test)
print("ac:", ac)


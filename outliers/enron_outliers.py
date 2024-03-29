#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("./tools/"))
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("./final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data_dict.pop("TOTAL",0)
data = featureFormat(data_dict, features)


### your code below

salary,bonus = zip(*data)
# try:
#     plt.plot(salary, bonus, color="blue")
# except NameError:
#     pass
plt.scatter(salary, bonus)
plt.show()

for d in data_dict:
    s = data_dict[d]['salary']
    if float(s) > 1000000:
        print(d, data_dict[d]['salary'],data_dict[d]['bonus'])
pass


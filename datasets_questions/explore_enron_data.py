#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib

enron_data = joblib.load(open("final_project/final_project_dataset.pkl", "rb"))
cnt = 0

for p in enron_data:
    if enron_data[p]['total_payments'] == "NaN" and enron_data[p]['poi'] == True: cnt +=1
    # if enron_data[p]['poi'] == True: cnt +=1
    # if enron_data[p]['total_payments'] == "NaN" : cnt +=1
# print(enron_data['LAY KENNETH L'.upper()])
print(cnt)
# cnt = 0
# for p in enron_data:
#     if enron_data[p]['email_address'] != "NaN": cnt +=1
# print(cnt)
print(len(enron_data), cnt/len(enron_data))


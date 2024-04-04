#!/usr/bin/python3

import os
import joblib
import re
import sys
import os
os.chdir(os.path.basename(os.path.dirname(__file__)))

sys.path.append(os.path.abspath("../tools/"))

words_file = "./your_word_data.pkl" 
authors_file = "./your_email_authors.pkl"
word_data = joblib.load( open(words_file, "rb"))
authors = joblib.load( open(authors_file, "rb") )

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit(word_data)
fnames = vectorizer.get_feature_names_out()
print(len(fnames))
print(fnames[34597])





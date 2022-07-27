# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 17:23:00 2022

@author: Mlad
"""
# --------------------------------------------------------
# data select #
# ----------- #

use_expanded = False # Without Wikipedia Data => False; With Wikipedia Data => True

# --------------------------------------------------------
# setup #
# ----- #

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import seaborn as sns
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

# --------------------------------------------------------
# data #
# -----#

saved_path = './saved/'

preprocessed_train_file_path = saved_path + 'pp_train_data.pkl'
preprocessed_test_file_path = saved_path + 'pp_test_data.pkl'

preprocessed_train_expanded_file_path = saved_path + 'pp_train_expanded_data.pkl'

trained_svm_model_file_path =  saved_path + 'trained_svm_model.sav'
trained_svm_model_expanded_file_path =  saved_path + 'trained_svm_expanded_model.sav'

print('loading preprocessed dataframes from disk...')

df_train = pd.read_pickle(preprocessed_train_file_path)
df_test = pd.read_pickle(preprocessed_test_file_path)

df_train_expanded = pd.read_pickle(preprocessed_train_expanded_file_path)

df_train = shuffle(df_train)
df_train_expanded = shuffle(df_train_expanded)

# --------------------------------------------------------
# config #
# ------ #

vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)

if use_expanded:
    train_expanded_vectors = vectorizer.fit_transform(df_train_expanded.text)
else:    
    train_vectors = vectorizer.fit_transform(df_train.text)
    
test_vectors = vectorizer.transform(df_test.text)

classifier_linear = svm.SVC(kernel='linear')

# --------------------------------------------------------
# training #
# -------- #

def save_model(model):
    if use_expanded:
        path = trained_svm_model_expanded_file_path        
    else:
        path = trained_svm_model_file_path
    pickle.dump(model, open(path, 'wb'))

def plot_matrix():
    if use_expanded:
        img_name = 'svm_expanded_result_matrix.png'
    else:
        img_name = 'svm_result_matrix.png'
    s = sns.heatmap(confusion_matrix(df_test['sentiment'], prediction_linear), annot = True, fmt='g', cmap="Oranges")
    s.set(xlabel='predicted', ylabel='real')
    # plt.savefig(img_name, format='png', dpi=1200)

if use_expanded:
    print('start expanded training...')
    classifier_linear.fit(train_expanded_vectors, df_train_expanded['sentiment'])
else:
    print('start training...')
    classifier_linear.fit(train_vectors, df_train['sentiment'])

save_model(classifier_linear)

# --------------------------------------------------------
# predicting #
# ---------- #

print('start predicting...')
prediction_linear = classifier_linear.predict(test_vectors)
report = classification_report(df_test['sentiment'], prediction_linear, output_dict=True)

print(report)

plot_matrix()

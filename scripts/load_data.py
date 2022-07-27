# -*- coding: utf-8 -*-
"""
created on sat jul  2 12:40:11 2022

@author: mlad
"""

# --------------------------------------------------------
# setup #
# ----- #

from datasets import load_dataset
import pandas as pd
import pickle

# --------------------------------------------------------
# config #
# ------ #

train_file_path = './saved/train_data.pkl'
test_file_path = './saved/test_data.pkl'
val_file_path = './saved/val_data.pkl'
pickle
# --------------------------------------------------------
# data loading #
# ------------ #

dataset = load_dataset('amazon_reviews_multi', 'de')

df_train = pd.DataFrame(data=dataset['train'], columns=dataset['train'].column_names)
df_test = pd.DataFrame(data=dataset['test'], columns=dataset['test'].column_names)
df_val = pd.DataFrame(data=dataset['validation'], columns=dataset['validation'].column_names)

# --------------------------------------------------------
# data saving #
# ------------#

df_train.to_pickle(train_file_path)
df_test.to_pickle(test_file_path)
df_val.to_pickle(val_file_path)

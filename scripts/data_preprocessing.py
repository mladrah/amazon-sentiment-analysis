# --------------------------------------------------------
# setup #
# ----- #

import pandas as pd
import emoji
import re
import string
import json
from HanTa import HanoverTagger as ht
import matplotlib.pyplot as plt

# --------------------------------------------------------
# config #
# ------ #

saved_path = './saved/'

train_file_path = saved_path + 'train_data.pkl'
test_file_path = saved_path + 'test_data.pkl'
val_file_path = saved_path + 'val_data.pkl'
neutral_file_path = saved_path + 'neutral_data.pkl'

preprocessed_train_file_path = saved_path + 'pp_train_data.pkl'
preprocessed_test_file_path = saved_path + 'pp_test_data.pkl'
preprocessed_val_file_path = saved_path + 'pp_val_data.pkl'
preprocessed_neutral_file_path = saved_path + 'pp_neutral_data.pkl'

preprocessed_train_expanded_file_path = saved_path + 'pp_train_expanded_data.pkl'

stop_words_path = saved_path + 'stop_words_german.json'
tagger = ht.HanoverTagger('morphmodel_ger.pgz')

# --------------------------------------------------------
# data #
# ---- #

print('loading dataframes from disk...')

df_train = pd.read_pickle(train_file_path)
df_test = pd.read_pickle(test_file_path)
df_val = pd.read_pickle(val_file_path)
df_neutral = pd.read_pickle(neutral_file_path)

with open(stop_words_path, encoding='utf-8') as f:
    stop_words = json.load(f)

# --------------------------------------------------------
# preprocessing #
# ------------- #

print('start data preprocessing...')

def clean_data(df):
    if 'review_title' in df:
        df["text"] = df["review_title"] + '. ' + df["review_body"] # combine title with body
        df = df.drop(['review_id','product_id','reviewer_id', # remove not needed columnsk
                      'product_category', 'language', 'review_body', 
                      'review_title'], axis=1, errors='ignore')
    df = df.dropna(how='any',axis=0) # drop rows with null value
    df = df[df['text'] != ''] # drop rows where text contains an empty string
    df = df[df['stars'] <= 5] # drop rows where star value is greater than 5
    df = df[df['stars'] >= 1] # drop rows where star value is smaller than 1
    return df

def remove_stop_words(text):
    clean_text = []
    words = text.split()
    for word in words:
        if word not in stop_words:
            clean_text.append(word)
    clean_text = ' '.join(clean_text)
    return clean_text

def lemmatize_text(text):
    lemmatized_text = []
    words = text.split()
    for word in words:
        lemmatized_text.append(tagger.analyze(word)[0])
    lemmatized_text = ' '.join(lemmatized_text)
    return lemmatized_text
    
def clean_text(df):
    url_pattern = re.compile(r'https?://\s+|www\.\s+')
    for index, row in df.iterrows():
        print(str(index) + ' / ' + str(len(df.index)) + ' : ' +str(index * 100 / len(df.index)))
        text = row['text']
        text = url_pattern.sub(r'', text) # remove urls
        text = re.sub('\s*@\s*\s?', '', text) # remove emails
        text = re.sub(r'[0-9]+', '', text) # remove numbers
        text = re.sub('\s+', ' ', text) # remove new line characters
        text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuations
        if emoji.emoji_count(text) > 0: # check if emoji is in text to store it for testing purposes 
            text = emoji.demojize(text, delimiters=('', ' '), language='de') # replaces emoji with text -> thumbs_up
        # text = text.lower() # to lower case to look for stop words
        # text = remove_stop_words(text)
        # text = lemmatize_text(text)
        # text = text.lower() # to lower case to lower lemmatized words
        text = re.sub(' +', ' ', text) # remove multiple white spaces
        df.at[index, 'text'] = text # replace old value with clean value
        # print('v: result\n' + text + '\n')
    return df

def label_by_stars(stars):
    if stars <= 2:
        return '0'
    elif stars == 3:
        return '1'
    else: 
        return '2'
    
def labeling(df):
    df['sentiment'] = df.apply(lambda row: label_by_stars(row.stars), axis=1) # class as new column
    df = df.drop(['stars'], axis=1, errors='ignore')
    return df

df_train = clean_data(df_train)
df_train = clean_text(df_train)
df_train = labeling(df_train)
df_train.to_pickle(preprocessed_train_file_path)

df_test = clean_data(df_test)
df_test = clean_text(df_test)
df_test = labeling(df_test)
df_test.to_pickle(preprocessed_test_file_path)

df_val = clean_data(df_val)
df_val = clean_text(df_val)
df_val = labeling(df_val)
df_val.to_pickle(preprocessed_val_file_path)

df_neutral = clean_data(df_neutral)
df_neutral = clean_text(df_neutral)
df_neutral = labeling(df_neutral)
df_neutral.to_pickle(preprocessed_neutral_file_path)

# --------------------------------------------------------
# loading #
# ------- #

df_train = pd.read_pickle(preprocessed_train_file_path)
df_test = pd.read_pickle(preprocessed_test_file_path)
df_val = pd.read_pickle(preprocessed_val_file_path)
df_neutral = pd.read_pickle(preprocessed_neutral_file_path)

# df_train.to_csv('preprocessed_train_data.csv', sep=',', encoding='utf-8', index=False)

# --------------------------------------------------------
# expanding #
# --------- #

df_neutral_for_train = df_neutral[0:40000]

df_train_expanded = pd.concat([df_train, df_neutral_for_train])
df_train_expanded = df_train_expanded.reset_index(drop=True)
df_train_expanded.to_pickle(preprocessed_train_expanded_file_path)

# fig, ax = plt.subplots()
# width = 0.35  
# labels = ['0', '1', '2']
# amz = [80000,40000,80000]
# wiki = [80000,80000,80000]
# ax.bar(labels, wiki, width, label='Wikipedia data', color='#FF5900')
# ax.bar(labels, amz, width, label='Amazon data', color='orange')
# ax.set_ylabel('count')
# ax.set_xlabel('class')
# ax.legend(loc='lower right')
# plt.savefig('data_distribution.svg', format='svg', dpi=1200)
# plt.show()
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
import seaborn as sns
from sklearn.utils import shuffle
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import json

# --------------------------------------------------------
# data #
# -----#

saved_path = './saved/'

preprocessed_train_file_path = saved_path + 'pp_train_data.pkl'
preprocessed_val_file_path = saved_path + 'pp_val_data.pkl'
preprocessed_test_file_path = saved_path + 'pp_test_data.pkl'

preprocessed_train_expanded_file_path = saved_path + 'pp_train_expanded_data.pkl'

trained_bert_model_file_path =  saved_path + 'trained_bert_model_weights/weights'
trained_bert_model_expanded_file_path =  saved_path + 'trained_bert_model_expanded_weights/weights'

preds_file_path = saved_path + 'bert_preds_data.pkl'
preds_expanded_file_path = saved_path + 'bert_preds_expanded_data.pkl'

print('loading preprocessed dataframes from disk...')

df_train = pd.read_pickle(preprocessed_train_file_path)
df_val = pd.read_pickle(preprocessed_val_file_path)
df_test = pd.read_pickle(preprocessed_test_file_path)

df_train_expanded = pd.read_pickle(preprocessed_train_expanded_file_path)

df_train = shuffle(df_train)
df_val = shuffle(df_val)

df_train_expanded = shuffle(df_train_expanded)

# --------------------------------------------------------
# config #
# ------ #

# models: deepset/gbert-large OR dbmdz/bert-base-german-cased
tokenizer = BertTokenizer.from_pretrained("deepset/gbert-large")
model = TFBertForSequenceClassification.from_pretrained("deepset/gbert-large", num_labels=3)

# --------------------------------------------------------
# training #
# -------- #

max_length = 256

def save_model(model):
    if use_expanded:
        print('saving expanded model...')
        with open('./model_expanded_config.json', 'w') as f:
            json.dump(model.to_json(), f)
        model.save_weights(trained_bert_model_expanded_file_path)
    else:
        print('saving model...')
        with open('./model_config.json', 'w') as f:
            json.dump(model.to_json(), f)
        model.save_weights(trained_bert_model_file_path)

def plot_matrix():
    if use_expanded:
        img_name = 'bert_prediction_expanded_matrix.png'
    else:
        img_name = 'bert_prediction_matrix.png'
    s = sns.heatmap(confusion_matrix(df_test['sentiment'].astype(str), prediction_linear.astype(str)), annot = True, fmt='g', cmap="BuPu")
    s.set(xlabel='predicted', ylabel='real')
    # plt.savefig(img_name, format='png', dpi=1200)

# reference for bert input transformation: https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671

InputExample(guid=None,
             text_a = "Hello, world",
             text_b = None,
             label = 1)

def convert_data_to_examples(train, val, DATA_COLUMN, LABEL_COLUMN): 
  train_InputExamples = train.apply(lambda x: InputExample(guid=None,
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)

  validation_InputExamples = val.apply(lambda x: InputExample(guid=None, 
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)
  
  return train_InputExamples, validation_InputExamples

  train_InputExamples, validation_InputExamples = convert_data_to_examples(train, 
                                                                           val, 
                                                                           'text', 
                                                                           'sentiment')

def convert_test_to_examples(test, DATA_COLUMN, LABEL_COLUMN): 
  test_InputExamples = test.apply(lambda x: InputExample(guid=None, 
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)
  
  return test_InputExamples

  test_InputExamples = convert_test_to_examples(test, 'text', 'sentiment')
  
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] 

    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, 
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


DATA_COLUMN = 'text'
LABEL_COLUMN = 'sentiment'

if use_expanded:
    print('start expanded transforming...')
    train_InputExamples, validation_InputExamples = convert_data_to_examples(df_train_expanded, df_val, DATA_COLUMN, LABEL_COLUMN)
else:
    print('start configuring...')
    train_InputExamples, validation_InputExamples = convert_data_to_examples(df_train, df_val, DATA_COLUMN, LABEL_COLUMN)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

print('start training...')

model.fit(train_data, epochs=2, validation_data=validation_data)

save_model(model)

# --------------------------------------------------------
# predicting #
# ---------- #

print('start predicting...')
test_inputExamples = convert_test_to_examples(df_test, DATA_COLUMN, LABEL_COLUMN)
    
test_data = convert_examples_to_tf_dataset(list(test_inputExamples), tokenizer)
test_data = test_data.batch(32)

results = model.predict(test_data, verbose=True)
tf_predictions = tf.nn.softmax(results[0], axis=-1)
prediction_linear = tf.argmax(tf_predictions, axis=1)
prediction_linear = prediction_linear.numpy()
prediction_linear = pd.DataFrame(prediction_linear)

if use_expanded:
    prediction_linear.to_pickle(preds_expanded_file_path)
else:    
    prediction_linear.to_pickle(preds_file_path)

report = classification_report(df_test['sentiment'].astype(int), prediction_linear.astype(int), output_dict=True)
    
print(report)

plot_matrix()

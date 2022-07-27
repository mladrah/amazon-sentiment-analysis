# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 17:23:00 2022

@author: Mlad
"""
# --------------------------------------------------------
# setup #
# ----- #

import pandas as pd
import seaborn as sns
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# from tensorflow.compat.v2.experimental import dtensor

# --------------------------------------------------------
# data #
# -----#

saved_path = 'C:/Users/rahmi/Desktop/amazon sentiment analysis/saved/'
preprocessed_train_file_path = saved_path + 'pp_train_data.pkl'
preprocessed_test_file_path = saved_path + 'pp_test_data.pkl'
preprocessed_val_file_path = saved_path + 'pp_val_data.pkl'
trained_bert_model_file_path =  saved_path + 'trained_bert_model_weights/weights'
preds_file_path = saved_path + 'bert_preds_data.pkl'

print('loading preprocessed dataframes from disk...')

df_train = pd.read_pickle(preprocessed_train_file_path)
df_test = pd.read_pickle(preprocessed_test_file_path)
df_val = pd.read_pickle(preprocessed_val_file_path)

# --------------------------------------------------------
# config #
# ------ #

def load_model():
    model = TFBertForSequenceClassification.from_pretrained("dbmdz/bert-base-german-uncased", num_labels=3)
    model.load_weights(trained_bert_model_file_path)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])
    return model

tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
model = load_model()

# --------------------------------------------------------
# predicting #
# ---------- #

max_length = 128
batch_size = 6

def plot_matrix():
    s = sns.heatmap(confusion_matrix(df_val['sentiment'].astype(str), prediction_linear.astype(str)), annot = True, fmt='g', cmap="BuPu")
    s.set(xlabel='predicted', ylabel='real')
    plt.savefig('bert_prediction_matrix.png', format='png', dpi=1200)

InputExample(guid=None,
             text_a = "Hello, world",
             text_b = None,
             label = 1)

def convert_val_to_examples(val, DATA_COLUMN, LABEL_COLUMN): 
  val_InputExamples = val.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)
  
  return val_InputExamples

  val_InputExamples = convert_val_to_examples(val, 'text', 'sentiment')
  
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default # CHECK THIS for pad_to_max_length
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

print('start configuring...')

val_InputExamples = convert_val_to_examples(df_val, DATA_COLUMN, LABEL_COLUMN)

val_data = convert_examples_to_tf_dataset(list(val_InputExamples), tokenizer)
val_data = val_data.batch(32)

print('start predicting...')

results = model.predict(val_data, verbose=True)
tf_predictions = tf.nn.softmax(results[0], axis=-1)
prediction_linear = tf.argmax(tf_predictions, axis=1)
prediction_linear = prediction_linear.numpy()
prediction_linear = pd.DataFrame(prediction_linear)

report = classification_report(df_val['sentiment'].astype(int), prediction_linear.astype(int), output_dict=True)
print(report)

plot_matrix()

prediction_linear = pd.read_pickle(preds_file_path)
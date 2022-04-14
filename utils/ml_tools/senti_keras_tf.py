
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
import numpy
import re
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, GRU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from nltk.corpus import stopwords
from numpy import random
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from bs4 import BeautifulSoup
import urllib
from tikads import *

import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os

import pandas as pd
max_words = 500
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = BeautifulSoup(text, "lxml").text  # HTML decoding
    text = text.lower()  # lowercase text
    # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    # delete symbols which are in BAD_SYMBOLS_RE from text
    text = BAD_SYMBOLS_RE.sub('', text)
    # delete stopwors from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text


def senti_keras_tf(file_path, model_path):
    if(model_path == "initial"):
        model_path = "/home/tikaadmin/tikapam/tikapam_text/models/sentiment/keras_tf"
    else:
        return ("wrong model name")
    ann_data = "neutral"
    clafy_label = {}
    senti = {}
    ann_data = {}
    ann = {}
    output = {}
    json_file = open(model_path+'/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_path+"/model.h5")
    url_path = getData(file_path)
    data_file = urllib.urlopen(url_path)

    text1 = " "
    for line in data_file:
        text1 = text1+line
    words = text_to_word_sequence(text1)
    vocab_size = len(words)
    clear_test = clean_text(text1)
    result = one_hot(clear_test, round(vocab_size*1.3))
    result = numpy.array(result, dtype=None, copy=True,
                         order=None, subok=False, ndmin=2)
    result = sequence.pad_sequences(result, maxlen=max_words)
    predictions = loaded_model.predict_classes(result, verbose=0)
    if(predictions == 1):
        result = "possitive"
    else:
        result = "negative"

    file_name = url_path.split("/")
    clafy_label['classification_label'] = result
    senti['sentiment'] = clafy_label
    ann['annotation_type'] = "labeling"       
    ann['data_filename'] = file_name[-1]
    ann['data_type'] = 'text'
    ann['data_annotation'] = senti
    output['annotation'] = ann
    return(output)

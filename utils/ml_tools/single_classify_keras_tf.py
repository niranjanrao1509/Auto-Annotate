import os
import csv
import numpy as np
import pandas as pd
import json
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
import re
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import pickle
import urllib
from tikads import *


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
modelpath = "/home/tikaadmin/tikapam/tikapam_text/source_code/single_labeled/keras_tf"


def clean_text(text):
    text = BeautifulSoup(text, "lxml").text  # HTML decoding
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text


def single_classify_keras_tf(file_path, model_path):
    if(model_path == "initial"):
        model_path = "/home/tikaadmin/tikapam/tikapam_text/models/single labelled/keras_tf"
    ann_data = "neutral"
    clafy_label = {}
    senti = {}
    ann_data = {}
    ann = {}
    output = {}
    url_path = getData(file_path)
    data_file = urllib.urlopen(url_path)
    # data_file = data_file.read()
    # file = open(file_path, "r")
    text1 = " "
    for line in data_file:
        text1 = text1+line
    clear_text = clean_text(text1)
    with open('sample.csv', 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow([' ', 'posts'])
        filewriter.writerow(["0", text1])
    df = pd.read_csv('sample.csv')
    train_size = int(len(df) * .7)
    test_posts = df['posts'][:train_size]
    max_words = 1000
    tokenize = text.Tokenizer(num_words=max_words, char_level=False)
    tokenize.fit_on_texts(test_posts)
    x_test = tokenize.texts_to_matrix(test_posts)
    json_file = open(model_path+'/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(modelpath+"/model.h5")
    with open(modelpath+"/labels.pickle", "rb") as f:
        labels = pickle.load(f)
    result = loaded_model.predict_classes(x_test)
    encoder = LabelEncoder()
    encoder.fit(labels)
    result = encoder.inverse_transform(result)
    file_name = url_path.split("/")
    clafy_label['classification_label'] = result[0]
    senti['classification'] = clafy_label
    ann['annotation_type'] = "labeling"       
    ann['data_filename'] = file_name[-1]
    ann['data_type'] = 'text'
    ann['data_annotation'] = senti
    output['annotation'] = ann
    os.remove("sample.csv")
    return(output)

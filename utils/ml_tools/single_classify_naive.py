import os
import pickle
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from numpy import random
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from bs4 import BeautifulSoup
from misc import *
from tikads import *
import urllib
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


def single_classify_naive(filepath, model_path):
#    if(model_path == "initial"):
#        model_path = "MjE3MzEkNDoxNDc4ODcuNTYkMTo0OjE1NDA1NTUzNDEuMTE="
#    else:
#        return ("wrong model path")
    model_url="stackoverflow_nb.pickle"

#    url_path = getData(filepath)
#    print(model_url_path)

    ann_data = "neutral"
    clafy_label = {}
    senti = {}
    ann_data = {}
    ann = {}
    output = {}

    #model = pickle.load(urllib.urlopen(model_url))
    with open(model_url,'rb') as model:
        model = pickle.load(model)
    data_file = urllib.request.urlopen(filepath)
    text = data_file.read()
    text = text.decode('ascii', 'ignore')
    cleaned_text = clean_text(text)
    file = open("test.txt", "w")


#    data_file = urllib.urlopen(filepath)
#    data_file = data_file.read()
    # data_file = open(filepath, "r")
#    cleaned_text = clean_text(data_file)
#    file = open("test.txt", "w")

    file.write(cleaned_text)
    file.close()
    data_file = open("test.txt", "r")
    result = model.predict(data_file)
    print(result)

    file_name = filepath.split("/")

    clafy_label['classification_label'] = result[0]
    senti['classification'] = clafy_label
    ann['annotation_type'] = "labeling"       
    ann['data_filename'] = file_name[-1]
    ann['data_type'] = 'text'
    ann['data_annotation'] = senti
    output['annotation'] = ann
    os.remove("test.txt")
    return(output)

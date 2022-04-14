import pickle
import cPickle
import re
import math
import collections
import itertools
import os
import nltk
import nltk.classify.util
import nltk.metrics
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import stopwords
import urllib
from tikads import *
model_path = "/home/tikaadmin/tikapam/tikapam_text/source_code/nltk_sentiment"


def create_word_features(words):
    useful_words = [
        word for word in words if word not in stopwords.words("english")]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict


def senti_nltk(text_path, model_path):
    if(model_path == "initial"):
        model_path = "MjE3MzQkNDoxMjI1MTguMjUkNDo0OjE1NDA1NTUzNDAuNTE="
    else:
        return ("wrong model")
    model_url_path = getData(model_path)
    clafy_label = {}
    senti = {}
    ann_data = {}
    ann = ""
    ann = {}
    output = {}

    classifier = pickle.load(urllib.urlopen(model_url_path))
    f.close()
    url_path = getData(text_path)

    data_file = urllib.urlopen(url_path)
    text = data_file.read()

    text = text.decode('cp850')

    words = word_tokenize(text)
    words = create_word_features(words)
    ann_data = classifier.classify(words)
    file_name = url_path.split("/")
    clafy_label['classification_label'] = ann_data
    senti['sentiment'] = clafy_label
    ann['data_filename'] = file_name[-1]
    ann['data_type'] = 'text'
    ann['data_annotation'] = senti
    output['annotation'] = ann
    return(output)

# classify_text_textblob("sample_text_possitive.txt")

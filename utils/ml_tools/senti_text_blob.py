from textblob import TextBlob
import urllib
import requests
from tikads import *


def senti_textblob(filepath, model_path):
    ann_data = "neutral"
    clafy_label = {}
    senti = {}
    ann_data = {}
    ann = {}
    output = {}
    #url_path = getData(file_path)
    #data_file = urllib.urlopen(url_path)
    data_file = urllib.request.urlopen(filepath)
    text = data_file.read()
    s = text.decode('ascii', 'ignore')
    analysis = TextBlob(s)
    print(analysis.sentiment)
    # if analysis.sentiment.subjectivity <= 0.8:
    if (analysis.sentiment.polarity >= 0.075 or analysis.sentiment.polarity <= 0.075):
        if analysis.sentiment.polarity > 0.075:
            ann_data = "positive"
        else:
            ann_data = "negative"

        # else:
        #     ann_data = "subjective"
    print(ann_data)
    file_name = filepath.split("/")
    clafy_label['classification_label'] = ann_data
    senti['sentiment'] = clafy_label
    ann['annotation_type'] = "labeling"       
    ann['data_filename'] = file_name[-1]
    ann['data_type'] = 'text'
    ann['data_annotation'] = senti
    output['annotation'] = ann
    return(output)


# classify_text_textblob('sample_text_negative.txt')

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import urllib
from tikads import *
import time
analyzer = SentimentIntensityAnalyzer()


def senti_vadersentiment(filepath, model_path):
    clafy_label = {}
    senti = {}
    ann_data = {}
    ann = {}
    output = {}
    #url_path = getData(file_path)
    data_file = urllib.request.urlopen(filepath)
    text = data_file.read()
    # f = open(file_path, "r")
    # text = data_file.read()
    text = text.decode('ascii', 'ignore')
    vs = analyzer.polarity_scores(text)
    print('vaderrr')
    print(vs)
    if not vs['neg'] > 0:
        if vs['pos']-vs['neg'] > 0:
            ann_data = "positive"
    elif vs['compound'] <= 0:
        ann_data = "negative"
    else:
        ann_data="neutral"
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

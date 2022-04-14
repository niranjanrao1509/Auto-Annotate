import glob
import os
import librosa
import tensorflow as tf
import numpy as np
import soundfile as sf
import pandas as pd
import json
def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)

def parse_json_files(filename):
    labels =  np.empty(0,dtype=int)
    listOfFile = os.listdir(filename)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(filename, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    
    for f in allFiles:
        a=pd.read_csv(f)
        label = a.columns[3]
        if label == "air_conditioner":
            labels = np.append(labels,0)
        elif label == "car_horn":
            labels = np.append(labels,1)
        elif label == "children_playing":
            labels = np.append(labels,2)
        elif label == "dog_bark":
            labels = np.append(labels,3)
        elif label == "drilling":
            labels = np.append(labels,4)
        elif label == "engine_idling":
            labels = np.append(labels,5)
        elif label == "gun_shot":
            labels = np.append(labels,6)
        elif label == "jackhammer":
            labels = np.append(labels,7)
        elif label == "siren":
            labels = np.append(labels,8)
        elif label == "street_music":
            labels = np.append(labels,9)
    return labels


def extract_features_classify(filename,file_ext="*.wav",bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    sound_clip,s = sf.read(filename)
    sound_clip_float = np.asfarray(sound_clip)
    sound_clip_input = []
    for i in range(0,len(sound_clip_float)):
        sound_clip_input.append(sound_clip_float[i][0])
    print('firstdone')
    sound_clip_in = np.reshape(sound_clip_input,(len(sound_clip_input),)) 
    for (start,end) in windows(sound_clip_in,window_size):
        for i in range(0,200):
            if(len(sound_clip_in[start:end]) == window_size):
                signal = sound_clip_in[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                logspec = librosa.amplitude_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    features = features[0:1297,:]
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    return np.array(features)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = 10
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

def preprocessing_classify(filename,train_json,file_ext="*.wav"):
    features= extract_features_classify(filename)
    labels = parse_json_files(train_json)
    labels = one_hot_encode(labels)
    train_test_split = np.random.rand(len(labels)) < 1
    train_x = features[train_test_split]
    train_y = labels[train_test_split]
    return features, train_x, train_y


def classify_audio_rnn(filename,model_path):
    train_json = '/data/envs/tika-tools/tpm/Audio/UrbanSound/data/metacsv/'
    labels_list = ['air_conditioner', 'car_horn', 'chilren_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']
    sub_dirs = filename.split('/')[-2]
    temp = "/"
    parent_dir = temp.join(filename.split('/')[:-2])
    with tf.Session() as sess:
    # Restore variables from disk.
        saver = tf.train.import_meta_graph(model_path)
        saver.restore(sess, 'rnn_model.ckpt')
        print("Model restored.")

    y_ = tf.get_collection("y_")[0]
    Y = tf.get_collection("Y")[0]
    X = tf.get_collection("X")[0]
    init = tf.global_variables_initializer() 
    features, train_x, train_y = preprocessing_classify(filename,train_json,file_ext="*.wav")
    y_true, y_pred = None, None
    with tf.Session() as sess:
      sess.run(init)
      y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: train_x})
      y_true = sess.run(tf.argmax(train_y,1))
    temp = labels_list[y_pred[0]]
    item = {"annotation":{}}
    annot = {}
    ann['annotation_type'] = "labeling"       
    annot["data_filename"] = filename.split("/")[-1]
    annot["data_type"] = "audio"
    annot["data_annotation"] = {"audio_classification":[]}
    annot["data_annotation"]["audio_classification"].append({"label":temp})
    item["annotation"] = annot
    return item
import os
from os import system
from celery import Celery
from celery.utils.log import get_task_logger
from celery.signals import worker_init, worker_process_init
import json
from celery.concurrency import asynpool
from misc import *
import sys
sys.path.insert(0, TP_ML_SCRIPTS_PATH)
import wget
import urllib.request
from misc import *
from lndmrk_dlib import *
from bb_tf import *
from bb_aws import *
from classify_tf import *
from ss_tf import *
#from bb_fb import bb_detectron_main
#from classify_fb import classify_detectron_main
from classify_yolo import classify_yolo_main
from bb_yolo import bb_yolo_main
import urllib.request
from senti_text_blob import *
from senti_vadersentiment import *
from single_classify_log_reg import *
from single_classify_svm import *
from single_classify_naive import *
from modified_cnn import *
from modified_rnn import *
from random import randint

def random_with_N_digits(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)
	
asynpool.PROC_ALIVE_TIMEOUT = 100.0 

env=os.environ  
CELERY_BROKER_URL=env.get('CELERY_BROKER_URL','pyamqp://' + RBMQ_USERNAME+':' + RBMQ_PASSWORD+'@' + RBMQ_HOST_IPADDR +':' + str(RBMQ_PORT)+'/'+RBMQ_HOSTNAME),   
CELERY_RESULT_BACKEND=env.get('CELERY_RESULT_BACKEND','amqp')

app= Celery('tp_api',  
                broker=CELERY_BROKER_URL,
                backend=CELERY_RESULT_BACKEND)
app.conf.accept_content = ['pickle','json','application/text']
app.conf.task_serializer = 'json'

@app.task(ignore_result=True)
def print_debug(s):
    print (s)

def get_path(img_url):
	filename, file_extension = os.path.splitext(img_url)
	if(file_extension == '.jpg'):
		filename = str(random_with_N_digits(10))+ '.jpg'
	elif(file_extension == '.txt'):
		filename= str(random_with_N_digits(10)) + '.txt'
	else:
		filename = str(random_with_N_digits(10)) + '.wav'
	out_filepath= '/data/envs/tika-tools/tpm/tools/ml_tools/media/tmp_downloads/' + filename
	img_url=urllib.request.urlopen(img_url)
	with open(out_filepath,'wb') as output:
		output.write(img_url.read())
	img_url=out_filepath
	return img_url

@app.task(name='tikapam.classify')
def classify(img_url,model_name, topN, classifer_type=1):
	print('hello celery')
	img_url=getData(img_url)
	img_url=get_path(img_url)

	if classifer_type == 1:
		out = classify_tf_main(img_url, topN, model_name)
		os.remove(img_url)
		out = json.dumps(out)
		return (out)
	elif classifer_type == 2:
		print('innnnn')
		out = classify_yolo_main(img_url, topN, model_name)
		os.remove(img_url)
		out = json.dumps(out)
		return (out)
	elif classifer_type == 3:
		print('YOLOOO')
		print('No error')
		classify_detect = classify_detectron_main(img_url, topN, model_name)
		os.remove(img_url)
		return classify_detect
		
		print('error here')
	else:
		print('Invalid classifier type') 


@app.task(name='tikapam.bbox')
def bbox(img_url,model_name,topN,bbox_type=3):
	print('hello celery')
	img_url=getData(img_url)
	img_url=get_path(img_url)
	if bbox_type == 1:
		tf_model = Classify_bb(model_name)
		x = tf_model.bb_tf_main(img_url, topN)
		os.remove(img_url)
		return x
	elif bbox_type == 2:
		bb_model = bb_yolo_main(img_url, topN, model_name)
		os.remove(img_url)
		return bb_model
	elif bbox_type == 9:
		bb_model = bb_aws_main(img_url, topN, model_name)
		os.remove(img_url)
		return bb_model
  
	elif bbox_type == 3:
		bb_detectron = bb_detectron_main(img_url,topN)
		os.remove(img_url)
		return bb_detectron
	else:
		print('Invalid classifier type')


@app.task(name='tikapam.sseg')
def sseg(img_url, sseg_type=1):
	print('hello celery')
	img_url=getData(img_url)
	img_url=get_path(img_url)

	print('yoooo')
	if sseg_type == 1:
		print('1 it isss')
		out =  ss_tf_main(img_url)
		os.remove(img_url)

		return (out)	
	elif sseg_type == 2:
		out = ss_detectron_main(img_url,topN)
		os.remove(img_url)
		return(out)	
	else:
		print('Invalid classifier type') 


@app.task(name='tikapam.landmrk')
def landmrk(img_url,model_name,topN, lndmrk_type=1):
	print('hello celery')
	img_url=getData(img_url)
	img_url=get_path(img_url)

	if lndmrk_type == 3:
		out =  lndmrk_dlib_main(img_url,model_name)
		os.remove(img_url)
		return (out)	
	else:
		print('Invalid classifier type') 


@app.task(name='tikapam.classify_text')
def single_classify(path, model_name, classifer_type=1):
    path=getData(path)
    path=get_path(path)
    path='file://' + path

    if model_name == "svm":
        print(classifer_type)
        print(model_name)
        out = single_classify_svm(path, model_name)
        path= path.strip('file:').replace('///','/')
        os.remove(path)
        out = json.dumps(out)
        return (out)
    elif model_name == "naive":
        print('inside naiveee')
        print(model_name)
        out = single_classify_naive(path, model_name)
        path= path.strip('file:').replace('///','/')
        os.remove(path)
        out = json.dumps(out)
        return (out)

    elif model_name == "logreg":
        print(classifer_type)
        print(model_name)
        out = single_classify_log_reg(path, model_name)
        path= path.strip('file:').replace('///','/')
        os.remove(path)
        out = json.dumps(out)
        return (out)
    else:
        print('Invalid classifier type')

@app.task(name='tikapam.classify_audio')
def classifyaudio(path,model_name, classifer_type=1):
	path=getData(path)
	path=get_path(path)
	if model_name == "cnn":
		out = classify_audio_cnn(path, 'cnn_model.ckpt.meta')
		os.remove(path)
		out = json.dumps(out)
		return (out)
	elif model_name == "rnn":
		out = classify_audio_rnn(path, 'rnn_model.ckpt.meta')
		os.remove(path)
		out = json.dumps(out)
		return (out)
	else:
		print('Invalid classifier type') 


@app.task(name='tikapam.senti_text')
def senti_text(path, model_name, classifer_type=1):
    path=getData(path)
    path=get_path(path)
    path='file://' + path

    if model_name== "textblob":
        print('inside celery')
        print(classifer_type)
        print(model_name)
        out = senti_textblob(path, model_name)
        path= path.strip('file:').replace('///','/')
        os.remove(path)
        out = json.dumps(out)
        return (out)
    elif model_name== "vader":
        print('inside celery')
        print(classifer_type)
        print(model_name)
        out = senti_vadersentiment(path, model_name)
        path= path.strip('file:').replace('///','/')
        os.remove(path)
        out = json.dumps(out)
        return (out)
#    elif classifier_type == 8:
#        out = senti_vadersentiment(text_path, model_name)
#        return (out)
#    elif classifier_type == 9:
#        out = senti_keras_tf(text_path, model_name)
#        return (out)
#    else:
#        print('Invalid classifier type')

@app.task(name='tikapam.multi_text')
def multi_text(path, model_name, classifer_type=1):
    print(model_name)
    if model_name == "magpie":
        out = multi_classify_magpie(path, model_name)
        os.remove(path)
        out = json.dumps(out)
        return (out)
    if model_name == "spacy":
        out = multi_classify_spacy(text_path, model_name)
        os.remove(path)
        out = json.dumps(out)
        return (out)

    else:
        print('Invalid classifier type')


 





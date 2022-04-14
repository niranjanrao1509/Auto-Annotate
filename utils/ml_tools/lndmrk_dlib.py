import sys
import subprocess
import cv2
import numpy as np
from misc import *
TOOLS_PATH = '/data/envs/tika-tools/tpm/models'
TP_DLIB_DEFAULT_PATH = TOOLS_PATH +'/dlib'

def lndmrk_dlib_main(image_path,  model_name):
    if(model_name=='shape-pred'):
        MODEL_NAME_PATH = '/data/envs/tika-tools/tpm/models/dlib'
        MODEL_NAME = 'shape_predictor_68_face_landmarks'
    else:
        print('Model_name not found \n Try another model')
        return("Try another model")
    
    file_path= "--image_file" + image_path
    print(file_path)
    MODEL_FILE_PATH = MODEL_NAME_PATH
    MODEL_FILE = MODEL_NAME + '.dat'

    cmd = 'cd ' + MODEL_NAME_PATH + ';' + 'python lndmark.py ' + image_path 
    out = subprocess.check_output(cmd, shell=True)
    x= out.decode('utf-8')
    x=x.replace("'","\"")
    return x















    


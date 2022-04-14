import requests
import base64
from .annoformats import convert 
from .annoformats import constants as const
from .constants import ClassifierModel, MLToolType, AnnotationFormat, DataType
from .misc import *
import json
import google
import google.cloud
from google.cloud import vision
from google.cloud.vision import types
from oauth2client.client import GoogleCredentials
import boto3
import codecs
from botocore.exceptions import ClientError, ParamValidationError
from botocore.exceptions import ClientError, ParamValidationError
import io
import os
import time

def classify(data_path, data_type, classifier_type,classifier_model,annotation_type,topN):

    internal_api_flag = True
    if (data_type == DataType.IMAGE):
        path = base64.b64encode(data_path.encode("utf-8"))
        path_enc = path.decode('cp850')
        rsp = " "
        if(classifier_type == MLToolType.INT_IMG_YOLO):
            if(classifier_model == ClassifierModel.INT_IMG_YOLO_EXTRACTION):
                model_name = ClassifierModel.INT_IMG_YOLO_EXTRACTION.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/classify/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)+ '/' + str(data_type.value) 
                rsp = requests.get(req_url)    
            elif(classifier_model == ClassifierModel.INT_IMG_YOLO_RESNET152):
                model_name = ClassifierModel.INT_IMG_YOLO_RESNET152.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/classify/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)+ '/' + str(data_type.value) 
                rsp = requests.get(req_url)    
            elif(classifier_model == ClassifierModel.INT_IMG_YOLO_DARKNET):
                model_name = ClassifierModel.INT_IMG_YOLO_DARKNET.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/classify/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)+ '/' + str(data_type.value) 
                rsp = requests.get(req_url)    
            
            else:
                print("wrong classifier model")

        elif(classifier_type == MLToolType.INT_IMG_TF):
            if(classifier_model == ClassifierModel.IMG_YOLO_DARKNET or classifier_model == ClassifierModel.IMG_TF_INSCP):
                model_name = ClassifierModel.IMG_TF_INSCP.value
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/classify/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)+ '/' + str(data_type.value) 
                rsp = requests.get(req_url)    
            else:
                print(classifier_type.name +
                      " do not use " + classifier_model.name)

        else:
            print("wrong classifier")

    elif (data_type ==DataType.TEXT):
        path = base64.b64encode(data_path.encode("utf-8"))
        path_enc = path.decode('cp850')
        rsp = " "
        if(classifier_type == MLToolType.INT_TXT_SINGLE_LABEL):
            if(classifier_model == ClassifierModel.INT_TXT_SVM):
                model_name = ClassifierModel.INT_TXT_SVM.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/classify/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)+ '/' + str(data_type.value) 
                rsp = requests.get(req_url)    
            elif(classifier_model == ClassifierModel.INT_TXT_NAIVE):
                model_name = ClassifierModel.INT_TXT_NAIVE.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/classify/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)+ '/' + str(data_type.value) 
                rsp = requests.get(req_url)    
            elif(classifier_model == ClassifierModel.INT_TXT_LOG_REG):
                model_name = ClassifierModel.INT_TXT_LOG_REG.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/classify/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)+ '/' + str(data_type.value) 
                rsp = requests.get(req_url)    
            else:
                print("wrong classifier model")

        elif(classifier_type== MLToolType.INT_TXT_SENTIMENT):
            if(classifier_model == ClassifierModel.INT_TXT_TEXTBLOB):
                model_name = ClassifierModel.INT_TXT_TEXTBLOB.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/classify/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)+ '/' + str(data_type.value) 
                rsp = requests.get(req_url)    
            elif(classifier_model == ClassifierModel.INT_TXT_VSENTIMENT):
                model_name = ClassifierModel.INT_TXT_VSENTIMENT.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/classify/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)+ '/' + str(data_type.value) 
                rsp = requests.get(req_url)    


        elif(classifier_type== MLToolType.TXT_MULTI_LABEL):
            if(classifier_model == ClassifierModel.INT_TXT_MAGPIE):
                model_name = ClassifierModel.INT_TXT_MAGPIE.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/classify/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)+ '/' + str(data_type.value) 
                rsp = requests.get(req_url)    
            elif(classifier_model == ClassifierModel.INT_TXT_VSENTIMENT):
                model_name = ClassifierModel.INT_TXT_VSENTIMENT.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/classify/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)+ '/' + str(data_type.value) 
                rsp = requests.get(req_url)    
            else:
                print('Incorrect model')        

    elif (data_type == DataType.AUDIO):
        path = base64.b64encode(data_path.encode("utf-8"))
        path_enc = path.decode('cp850')
        rsp = " "
        if(classifier_type == MLToolType.INT_AUD_CLF):
            if(classifier_model == ClassifierModel.INT_AUD_CNN):
                model_name = ClassifierModel.INT_AUD_CNN.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/classify/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)+ '/' + str(data_type.value) 
                rsp = requests.get(req_url)    
                rsp = requests.get(req_url)

            elif(classifier_model == ClassifierModel.INT_AUD_RNN):
                model_name = ClassifierModel.INT_AUD_RNN.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/classify/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)+ '/' + str(data_type.value) 
                rsp = requests.get(req_url)    
                rsp = requests.get(req_url)

            else:
                print("wrong classifier model")
        else:
                print('Incorrect model')        

        
    else:
        print("only image and text Data Type")

    if(rsp == " "):
        return (" ")
    else:
        if(internal_api_flag):
            if(rsp.status_code == 200):
                rsp = rsp.text
                rsp = json.loads(rsp)
                if(annotation_type == AnnotationFormat.TIKA_XML):
                    return convert(const.AnnotationOutputFormat.TIKA_JSON,
                                   const.AnnotationOutputFormat.TIKA_XML, const.AnnotationOutputMode.OBJECT, rsp)
                elif(annotation_type.name == AnnotationFormat.TIKA_JSON.name):
                    return rsp
                elif(annotation_type.name == AnnotationFormat.YOLO_TXT.name):
                    return convert(const.AnnotationOutputFormat.TIKA_JSON,
                                   const.AnnotationOutputFormat.YOLO_TXT, const.AnnotationOutputMode.OBJECT, rsp)
                elif(annotation_type == AnnotationFormat.COCO_JSON):
                    return convert(const.AnnotationOutputFormat.TIKA_JSON,
                                   const.AnnotationOutputFormat.COCO_JSON, const.AnnotationOutputMode.OBJECT, rsp)
                elif(annotation_type == AnnotationFormat.PASCALVOC_XML):
                    return convert_clas_pasc(const.AnnotationOutputFormat.TIKA_JSON,
                                   const.AnnotationOutputFormat.PASCALVOC_XML, const.AnnotationOutputMode.OBJECT, rsp)
                else:
                    print('Invalid annotation format')
            elif(rsp.status_code == 500):
                print("Problem with the server")
            elif(rsp.status_code == 400):
                print("bad request")
            elif(rsp.status_code == 404):
                print(" requested file not found. Plz check the url ")
            else:
                print("error with status code ", rsp.status_code)
        else:
            if(annotation_type == AnnotationFormat.TIKA_XML):
                return convert(const.AnnotationOutputFormat.TIKA_JSON,
                               const.AnnotationOutputFormat.TIKA_XML, const.AnnotationOutputMode.OBJECT, rsp)
            elif(annotation_type.name == AnnotationFormat.TIKA_JSON.name):
                return rsp
            elif(annotation_type.name == AnnotationFormat.YOLO_TXT.name):
                return convert(const.AnnotationOutputFormat.TIKA_JSON,
                               const.AnnotationOutputFormat.YOLO_TXT, const.AnnotationOutputMode.OBJECT, rsp)
            elif(annotation_type == AnnotationFormat.COCO_JSON):
                return convert(const.AnnotationOutputFormat.TIKA_JSON,
                               const.AnnotationOutputFormat.COCO_JSON, const.AnnotationOutputMode.OBJECT, rsp)
            elif(annotation_type == AnnotationFormat.PASCALVOC_XML):
                return convert(const.AnnotationOutputFormat.TIKA_JSON,
                               const.AnnotationOutputFormat.PASCALVOC_XML, const.AnnotationOutputMode.OBJECT, rsp)
            else:
                print('Invalid annotation format')

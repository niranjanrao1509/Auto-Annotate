import requests
import base64
from .annoformats import convert
from .annoformats import constants as const
from .constants import ObjectDetectionModel, MLToolType, AnnotationFormat, DataType
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

def bbox(data_path, data_type, classifier_type=MLToolType.INT_IMG_TF,
         classifier_model=ObjectDetectionModel.IMG_TF_RESNET50,
         annotation_type=AnnotationFormat.TIKA_JSON, topN=1):
    rsp = " "
    non_tikapam = False
    if (data_type == DataType.IMAGE):
        path = base64.b64encode(data_path.encode("utf-8"))
        path_enc = path.decode('cp850')
        rsp = " "
        if(classifier_type == MLToolType.INT_IMG_YOLO):
            if(classifier_model == ObjectDetectionModel.IMG_TF_RESNET50 or classifier_model == ObjectDetectionModel.IMG_YOLO_V3):
                model_name = ObjectDetectionModel.IMG_YOLO_V3.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/bbox/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)
                rsp = requests.get(req_url)
                
            elif(classifier_model == ObjectDetectionModel.IMG_YOLO_V3TINY):
                model_name = ObjectDetectionModel.IMG_YOLO_V3TINY.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/bbox/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)
                rsp = requests.get(req_url)
            else:
                print("wrong classifier model")

        elif(classifier_type == MLToolType.INT_IMG_TF):
            if(classifier_model == ObjectDetectionModel.IMG_TF_RESNET50):
                model_name = ObjectDetectionModel.IMG_TF_RESNET50.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/bbox/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)
                rsp = requests.get(req_url)
                rsp = rsp.text
                return rsp
                
            elif(classifier_model == ObjectDetectionModel.IMG_TF_RESNET101):
                model_name = ObjectDetectionModel.IMG_TF_RESNET101.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/bbox/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)
                rsp = requests.get(req_url)
                rsp = rsp.text
                return rsp
                
            elif(classifier_model == ObjectDetectionModel.IMG_TF_INSCP_SSD):
                model_name = ObjectDetectionModel.IMG_TF_INSCP_SSD.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/bbox/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)
                rsp = requests.get(req_url)
                rsp = rsp.text
                return rsp

            elif(classifier_model == ObjectDetectionModel.IMG_TF_MOBI_NET):
                model_name = ObjectDetectionModel.IMG_TF_MOBI_NET.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/bbox/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)
                rsp = requests.get(req_url)
                rsp = rsp.text
                return rsp

            elif(classifier_model == ObjectDetectionModel.IMG_TF_INSCP_MASK):
                model_name = ObjectDetectionModel.IMG_TF_INSCP_MASK.value
                print(model_name)
                model_name = model_name + "=="
                model_encode = base64.b64encode(model_name.encode("utf-8"))
                model_encode = model_encode.decode('cp850')
                req_url = 'http://' + TP_API_ENGINE_IPADDR + ':' + TP_API_ENGINE_PORT + '/bbox/' + \
                    str(path_enc) + '/' + str(model_encode) + '/' + \
                    str(topN) + "/" + str(classifier_type.value)
                rsp = requests.get(req_url)
                rsp = rsp.text
                return rsp

            else:
                print(classifier_type.name +
                      " do not use " + classifier_model.name)
        else:
            print("wrong classifier")
    else:
        print("Unsupported data type")

    if(rsp == " "):
        return (" ")
    else:
        if(not non_tikapam):
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
                    return convert(const.AnnotationOutputFormat.TIKA_JSON,
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
            return (rsp)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  
import glob
import logging
import os
import sys
import time
import json
import inspect
import numpy as np
from misc import * 

#from caffe2.python import workspace
TOOLS_PATH = '/data/envs/tika-tools/tpm/models'
TP_DETECTRON_OBJ_DIR = TOOLS_PATH + '/detectron'

sys.path.insert(0, TP_DETECTRON_OBJ_DIR)

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis_fb_classify_bb as vis_utils

c2_utils.import_detectron_ops()

cv2.ocl.setUseOpenCL(False)

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=TP_DETECTRON_DEFAULT_MODEL_CONFIG,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=TP_DETECTRON_DEFAULT_MODEL_WTS,
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default=TP_DETECTRON_DEFAULT_IMG_EXT,
        type=str
    )
    return parser.parse_args()


def bb_detectron_main(image_path, N):
    args = parse_args()
    merge_cfg_from_file(args.cfg)
    if cfg.NUM_GPUS == 1:
        pass
    else:
        cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    im = cv2.imread(image_path)
    timers = defaultdict(Timer)
    t = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, im, None, timers=timers
        )
        boxes,segms,keyps,classes = vis_utils.convert_from_cls_format(cls_boxes, cls_segms, cls_keyps)
        score = boxes[:,-1]
        cls_str=[]
        
        for i in range(len(classes)):
            temp = vis_utils.get_class_string(classes[i],score[i],dummy_coco_dataset)
            cls_str.append(temp)
        cls_strarr = np.asarray(cls_str)
        cls_strarr=cls_strarr[:,np.newaxis]
        mix=np.hstack((boxes,cls_strarr))
        mix=mix[mix[:,4].argsort()][::-1]      

        ann_data={}
        ann={}
        out={}

        if N>len(classes):
            N=len(classes)

        if float(mix[N-1][4]) >= TF_BB_CONFIDENCE_THRES:     
            n=N
        else:
            n=len(classes)

        for i in range(n):
            if float(mix[i][4]) >= TF_BB_CONFIDENCE_THRES :
                label_data={}
                label_data['classification_label']=str(mix[i][5])
                label_data['point_2D']=str(round(float(boxes[i][0]),2))+','+str(round(float(boxes[i][1]),2)) , str(round(float(boxes[i][2]),2))+','+str(round(float(boxes[i][3]),2))                    
                ann_data.setdefault('bounding_box', []).append(label_data)

        ann['data_filename']=image_path
        ann['data_type']='image'
        ann['data_annotation']=ann_data
        out['annotation']=ann

        return out

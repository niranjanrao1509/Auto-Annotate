from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#TOOLS_PATH = '/data/envs/tika-tools/tpm/models'
#TP_TF_DEEPLAB_PATH = TOOLS_PATH+ '/deeplabv-3'

import argparse
import os
import sys
import numpy as np
import json
import inspect
import tensorflow as tf
from misc import * 
sys.path.insert(0, TP_TF_DEEPLAB_PATH)

import deeplab_model
from utils import preprocessing
from utils import dataset_util

from PIL import Image
import matplotlib

import matplotlib.pyplot as plt
plt.switch_backend('agg')

from tensorflow.python import debug as tf_debug


label_map = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
             'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default=TP_TF_SEMSEG_DEFAULT_DATA_PATH,
                    help='The directory containing the image data.')

parser.add_argument('--model_dir', type=str, default=TP_TF_SEMSEG_DEFAULT_MODEL_PATH,
                    help="Base directory for the model. "
                         "Make sure 'model_checkpoint_path' given in 'checkpoint' file matches "
                         "with checkpoint name.")

parser.add_argument('--base_architecture', type=str, default=TP_TF_SEMSEG_DEFAULT_MODEL,
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--output_stride', type=int, default=TP_TF_SEMSEG_DEFAULT_OUTPUT_STRIDE,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')


def ss_tf_main(image_path):
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    FLAGS, unparsed = parser.parse_known_args()
    pred_hooks = None
    if FLAGS.debug:
        debug_hook = tf_debug.LocalCLIDebugHook()
        pred_hooks = [debug_hook]

    model = tf.estimator.Estimator(
        model_fn=deeplab_model.deeplabv3_plus_model_fn,
        model_dir=FLAGS.model_dir,
        params={
            'output_stride': FLAGS.output_stride,
            'batch_size': 1,
            'base_architecture': FLAGS.base_architecture,
            'pre_trained_model': None,
            'batch_norm_decay': None,
            'num_classes': TP_TF_SEMSEG_NUM_CLASSES,
        })

    predictions = model.predict(
        input_fn=lambda: preprocessing.eval_input_fn([image_path]),
        hooks=pred_hooks)

    for pred_dict, image_path in zip(predictions, [image_path]):

        classes = pred_dict['classes']
        cls_unq = np.unique(classes)

        [p, q] = [classes.shape[0], classes.shape[1]]
        point = []
        for k in range(len(cls_unq)):
            point.append([])

        ann_data = {}
        ann = {}
        out = {}

        for i in range(p):
            for j in range(q):
                for k in range(1, len(cls_unq)):
                    if classes[i][j] == cls_unq[k]:
                        point[k].append(str(j) + ',' + str(i))

        for k in range(1, len(cls_unq)):
            label_data = {}
            label_data['classification_label'] = label_map[cls_unq[k]]
            label_data['point_2D'] = point[k]
            ann_data.setdefault('semantic_segmentation', []).append(label_data)

        ann['annotation_type'] = "labeling"       
        ann['data_filename'] = image_path
        ann['data_type'] = 'image'
        ann['data_annotation'] = ann_data
        out['annotation'] = ann
        return out

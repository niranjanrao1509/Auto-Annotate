#rabbitMQ 
RBMQ_HOSTNAME = 'tikahost'
RBMQ_HOST_IPADDR = '10.16.18.50'
RBMQ_PORT = 0
RBMQ_USERNAME = 'tikaadmin1'
RBMQ_PASSWORD = 'tika123'

TOOLS_PATH = '/data/envs/tika-tools/tpm/models'

# sem_seg_detectron
TP_TF_DEEPLAB_PATH = TOOLS_PATH+ '/deeplabv-3'

# sem_seg_tf
deeplab_dir = '/home/tikaadmin/deeplab'
TP_TF_SEMSEG_DEFAULT_DATA_PATH = TOOLS_PATH + '/deeplab/data_dir/VOCdevkit/VOC2012/JPEGImages'
TP_TF_SEMSEG_DEFAULT_MODEL_PATH = TOOLS_PATH + '/deeplab/model_dir/model'
TP_TF_SEMSEG_DEFAULT_MODEL = 'resnet_v2_101'
TP_TF_SEMSEG_DEFAULT_OUTPUT_STRIDE = 16
TP_TF_SEMSEG_NUM_CLASSES = 21



#ML scripts
TP_ML_SCRIPTS_PATH = '/data/envs/tika-tools/tpm/tools/ml_tools/'
TP_BB_TF_PATH = '/data/envs/tika-tools/tpm/models/models_obj_det/research'

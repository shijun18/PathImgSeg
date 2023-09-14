from utils import get_weight_path,print_params

__net__ = ['unet','unet++','deeplabv3+','pspnet','HDenseFormer_2D_16','HDenseFormer_2D_32']
__encoder_name__ = [None,'resnet18','resnet50']


data_path = {
    # competition
    'MoNuSeg':{
            'train_path':'/acsa-med/pathology/MoNuSeg/MoNuSeg/Train_Folder/hdf5/',
            'val_path':'/acsa-med/pathology/MoNuSeg/MoNuSeg/Val_Folder/hdf5/'
        }
}

keys = {
    'MoNuSeg':('image','label')
}

channel = {
    'MoNuSeg':3
}

roi_number = {
    'MoNuSeg':None,
}

num_class = {
    'MoNuSeg':2,
}

input_shape = {
    'MoNuSeg':(512,512),
}
####### config ####### 
DEVICE = '5' # device

DATASET = 'MoNuSeg'
NET_NAME = 'unet'
ENCODER_NAME = 'resnet18'
VERSION = 'v1.1'

# True if use internal pre-trained model
# Must be True when pre-training and inference
PRE_TRAINED = False
# True if use external pre-trained model 
EX_PRE_TRAINED = False
# True if use resume model
CKPT_POINT = False
FOLD_NUM = 5
# [1-FOLD_NUM]
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))


# Arguments for trainer initialization
#--------------------------------- single or multiple
CHANNEL = channel[DATASET]
ROI_NUMBER = roi_number[DATASET] # or 1,2,...
NUM_CLASSES = num_class[DATASET] 
PATH_DIR = data_path[DATASET]
INPUT_SHAPE = input_shape[DATASET]
#---------------------------------

#--------------------------------- others
BATCH_SIZE = 2 
CKPT_PATH = './ckpt/{}/fold{}'.format(VERSION,str(CURRENT_FOLD))
print(CKPT_PATH)
WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)



INIT_TRAINER = {
  'net_name':NET_NAME,
  'encoder_name':ENCODER_NAME,
  'lr':1e-3, #2d
  'n_epoch':100,
  'channels':CHANNEL,
  'num_classes':NUM_CLASSES, 
  'roi_number':ROI_NUMBER, 
  'input_shape':INPUT_SHAPE,
  'crop':0,
  'batch_size':BATCH_SIZE,
  'num_workers':4,
  'device':DEVICE,
  'pre_trained':PRE_TRAINED,
  'ex_pre_trained':EX_PRE_TRAINED,
  'ckpt_point':CKPT_POINT,
  'weight_path':WEIGHT_PATH,
  'weight_decay': 0.0001,
  'momentum':0.9,
  'gamma':0.1,
  'milestones':[50,80],
  'T_max':5,
  'topk':10,  
  'use_fp16':False,
  'transform_2d':[1,2,6,7,10],  
  'transformer_depth':24, #[8,12,24,36]
  'key_touple':keys[DATASET]
 }
print_params(INIT_TRAINER)
#---------------------------------

__loss__ = ['Cross_Entropy','DiceLoss','TopKLoss','CEPlusDice','FocalLoss','FLPlusDice']

LOSS_FUN = 'FocalLoss' if NUM_CLASSES == 2 else 'CEPlusDice'
SETUP_TRAINER = {
  'output_dir':'./ckpt/{}/{}'.format(DATASET,VERSION),
  'log_dir':'./log/{}/{}'.format(DATASET,VERSION),
  'optimizer':'Adam',
  'loss_fun':LOSS_FUN,
  'class_weight':None,
  'lr_scheduler':'poly_lr',
  'use_ds':'DenseFormer' in NET_NAME
  }
print_params(SETUP_TRAINER)
#---------------------------------

TEST_PATH  = None
# SAVE_PATH = './segout/{}_3d/fold{}'.format(VERSION,CURRENT_FOLD)



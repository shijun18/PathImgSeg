import os
import argparse
from trainer import SemanticSeg
import glob

from config import INIT_TRAINER, CHANNEL, INPUT_SHAPE,SETUP_TRAINER, CURRENT_FOLD, FOLD_NUM, PATH_DIR
from utils import count_params_and_macs
import time



def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--mode',
                        default='train-cross',
                        choices=["train", 'train-cross'],
                        help='choose the mode',
                        type=str)
    args = parser.parse_args()

    # Set data path & segnetwork
    if args.mode != 'train-cross':
        segnetwork = SemanticSeg(**INIT_TRAINER)
    #     print(get_parameter_number(segnetwork.net))
    #     print('params and gflops:')
    #     print(count_params_and_macs(segnetwork.net.cuda(),(1,CHANNEL) + INPUT_SHAPE))

    # Training
    ###############################################
    if args.mode == 'train-cross':
        for current_fold in range(1, FOLD_NUM + 1):
            print("=== Training Fold ", current_fold, " ===")
            segnetwork = SemanticSeg(**INIT_TRAINER)
            print(get_parameter_number(segnetwork.net))
            print('params and gflops:')
            print(count_params_and_macs(segnetwork.net.cuda(),(1,CHANNEL) + INPUT_SHAPE))
            train_path = glob.glob(os.path.join(PATH_DIR['train_path'],'*.hdf5'))
            val_path = glob.glob(os.path.join(PATH_DIR['val_path'],'*.hdf5'))
            
            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['cur_fold'] = current_fold
            start_time = time.time()
            segnetwork.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time() - start_time))


    if args.mode == 'train':
        train_path = glob.glob(os.path.join(PATH_DIR['train_path'],'*.hdf5'))
        val_path = glob.glob(os.path.join(PATH_DIR['val_path'],'*.hdf5'))
        
        print(train_path)
        
        SETUP_TRAINER['train_path'] = train_path
        SETUP_TRAINER['val_path'] = val_path
        SETUP_TRAINER['cur_fold'] = CURRENT_FOLD
		
        start_time = time.time()
        segnetwork.trainer(**SETUP_TRAINER)

        print('run time:%.4f' % (time.time() - start_time))
    ###############################################


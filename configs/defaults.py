import os
import time

from fvcore.common.config import CfgNode

#Init configurations
_C = CfgNode()

#Model Options
_C.MODEL = CfgNode()
# _C.MODEL.PRETRAIN_PATH = "/export/home/aneezahm001/nndl_2/checkpoints/celeba_age_gender"
_C.MODEL.PRETRAIN_PATH = None #Path to pretrain model directory for transfer learning
_C.MODEL.INPUT_SHAPE = (227, 227, 3)
_C.MODEL.DROPOUT = 0.5
_C.MODEL.INITIALIZER = "levi_hassner" #use "levi_hassner" to initialize the model weights using the levi hassner method 

#Optimizer configurations
_C.MODEL.OPTIMIZER = CfgNode()
_C.MODEL.OPTIMIZER.NAME = "sgd" #Name of the optimizer which will be used when loading the optimizer function
_C.MODEL.OPTIMIZER.ALPHA = 1e-2 #learning rate of the optimizer
_C.MODEL.OPTIMIZER.GAMMA = 5e-4 #Weight decay to apply to the optimizer
_C.MODEL.OPTIMIZER.BETAS = (0.9, 0.999) #Beta values to be used with the adam optimizer. This configuration is not required for sgd
_C.MODEL.OPTIMIZER.MOMENTUM = 0. #Momentum for SGD. Does not apply to adam.

#Training options
_C.TRAIN = CfgNode()
_C.TRAIN.ENABLE = True #Enable model training
_C.TRAIN.MAX_EPOCHS = 15 #Number of epochs to train the model
_C.TRAIN.CROSS_VALIDATE = True #Enable 5-fold cross validation as described in the levi-hassner paper

#Dataset options
_C.DATA = CfgNode()
_C.DATA.DATA_DIR = "/export/home/aneezahm001/nndl_2/adience/adience_age_gender/" #Path to the base directory containing the dataset files
_C.DATA.BATCH_SIZE = 8 #Training and testing batch size
_C.DATA.TARGET_LABEL = "both" #(age, gender, both) to specify the classification task to be carried out

_C.LOG_DIR = "/export/home/aneezahm001/nndl_2/age_gender/logs/adience" 
_C.RUN_SEED = 1 #Use this seed to make the code execution deterministic. Execusions with the same RUN_SEED will have the same results

_C.PREFIX = "" #Set internally

def assert_and_infer_cfg(cfg):
    log_folder = str(round(time.time())) + "_" + cfg.DATA.TARGET_LABEL

    if cfg.MODEL.PRETRAIN_PATH is not None:
        log_folder += "_celeba_pretrained"

    cfg.LOG_DIR = os.path.join(cfg.LOG_DIR, log_folder)
    cfg.LOG_FILE = os.path.join(cfg.LOG_DIR, "training.log.tsv")

    # cfg.TRAIN.CHECKPOINT_DIR = os.path.join(cfg.LOG_DIR, "train_files")

    if not os.path.exists(cfg.LOG_DIR):
        print(cfg.LOG_DIR, "does not exist. Creating it now...")    
        os.makedirs(cfg.LOG_DIR)
        # os.makedirs(cfg.TRAIN.CHECKPOINT_DIR)

    with open(cfg.LOG_FILE, "a") as af:
        write_string = str(cfg)
        af.write(write_string + "\n\n")
        print("Running with args \n", cfg)

    return cfg

def get_cfg():
    return _C.clone()
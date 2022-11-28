import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import random

from utils.logger import write_to_file
from configs.argparser import parse_args, load_config
from configs.defaults import assert_and_infer_cfg
from utils.run_training import run_model_training

def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def main():
    args = parse_args()
    cfg = load_config(args)
    cfg = assert_and_infer_cfg(cfg)

    set_seeds(cfg.RUN_SEED)
    set_global_determinism(cfg.RUN_SEED)

    if cfg.TRAIN.ENABLE:
        mean_accs = []

        if cfg.TRAIN.CROSS_VALIDATE:
            for i, test_split_folder in enumerate(os.listdir(cfg.DATA.DATA_DIR)):
                to_write = "Running on cross-validation fold %d" % (i+1)
                cfg.PREFIX = "fold_" + str(i+1)

                write_to_file(cfg.LOG_FILE, to_write)

                test_split_folder = os.path.join(cfg.DATA.DATA_DIR, test_split_folder)
                mean_accs.append(run_model_training(cfg, test_split_folder))

                to_write = "Mean accuracies = " + np.array2string(np.mean(np.array(mean_accs), axis=0)) + "\n\n"
                write_to_file(cfg.LOG_FILE, to_write)
                print(to_write)
        else:
            test_split_folder = os.path.join(cfg.DATA.DATA_DIR, os.listdir(cfg.DATA.DATA_DIR[0]))
            accs = run_model_training(cfg, test_split_folder)
            print("Validation accuracies over age and gender prediction :", accs)

    print("Runtime Complete.")
            

if __name__ == "__main__":
    main()
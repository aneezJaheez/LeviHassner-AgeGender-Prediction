# Simultaneous-Age-Gender-Prediction
I compare the performance of the single-task learning and multi-task learning method for the task of Age and Gender Prediction by inheriting and improving the Levi-Hassner model. The models are evaluated on the Adience dataset.

## Index
* [Overview](#Overview)
* [Dependencies and Environment](#Dependencies-and-Environment)
* [Datasets and Preparation](#Datasets-and-Preparation)
* [Running the Attack](#Running-the-Attack)
* [Visualizing the Results](#Visualizing-the-Results)

## Overview
This folder contains the model and associated python files to perform age and gender classification on the Adience benchmark dataset. The code is implemented using Python 3.8 and the models are implemented using TensorFlow 2.3. 

## Dependencies and Environment

The architecture is implemented using TensorFlow 2.10. A full list of dependencies are listed in the environment.yml file. Run the command below to install the required packages in a conda environment. 

```
conda env create -f environment.yml
```

## Datasets and Preparation
The datasets are prepared using a similar procedure followed in the paper [Age and Gender Classification using CNNs](https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.pdf). We laso use the same cross-validation procedure to evaluate the model.

To prepare the dataset, first you will need to preprocess the data using preproc.py. This assumes that there is a directory that is passed for an absolute directory, as well as a file containing a list of the training data images and the label itself, and the validation data, and test data if applicable. The preproc.py program generates 'shards' for each of the datasets, each containing JPEG encoded RGB images of size 256x256 and the associated age and gender labels.

```
python ./data/preproc.py --fold_dir ./adience/adience_age_gender/foldstrain_val_txt_files_per_fold/test_fold_is_0 --data_dir ./adience/aligned --output_dir ./adience/adience_age_gender/test_fold_is_0
```

The adience data_dir and fold_dir can be downloaded here: http://www.openu.ac.il/home/hassner/Adience/data.html

## Running the Attack

After installing the required dependencies and preparing the datasets for training the shadow models according to the recipe highlighted in the paper, the training can be run from the project directory of the via the command 

```
python run.py
```

In addition, the attack includes a list of configurations that can be tuned according to the attack requiremenets. A full list of configurations ar can be found in [configs/defaults.py]. The configurations can either be tuned directly in the file, or stated in the initial command to run the attack. For instance, to run model training for both age and gender recognition with a batch size of 50 for 20 epochs using the adam optimizer using a pretrained model and with a runtime seed of 100:

```
python run.py DATA.TARGET_LABEL both DATA.BATCH_SIZE 50 MODEL.OPTIMIZER.NAME adam TRAIN.MAX_EPOCHS 20 MODEL.PRETRAIN_PATH /export/home/aneezahm001/nndl_2/checkpoints/celeba_age_gender RUN_SEED 100
```

The configurations listed on the command line in the above manner take precendence over the configurations in the main configurations file.

## Visualizing the Results
On every run, the results are logged in the directory "./age_gender/logs". The logs contain the output over the entire training run in the "training.log.tsv" file which can be viewed as a txt file. The logs also contain tensorboard visualizations of the training dataset, validation dataset, and training and validation accuracies and losses over each cross-validation fold in the dataset. These logs can be visualized using tensorboard. For instance, to view the training logs for the task of age and gender recognition, use the following command from the project directory:


```
tensorboard --logdir=./age_gender/logs/adience/1668097217_both
```

Similarly, to view the training visualizations for the task of gender classification:

```
tensorboard --logdir=./age_gender/logs/adience/1668091466_gender
```

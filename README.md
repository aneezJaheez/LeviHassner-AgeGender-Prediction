# Simultaneous-Age-Gender-Prediction
I compare the performance of the single-task learning and multi-task learning method for the task of Age and Gender Prediction by inheriting and improving the Levi-Hassner model. The models are evaluated on the Adience dataset.

## Index
* [Multi-Task Learning](#Multi-Task-Learning)
* [Repository Overview](#Overview)
* [Dependencies and Environment](#Dependencies-and-Environment)
* [Datasets and Preparation](#Datasets-and-Preparation)
* [Running the Model](#Running-the-Model)
* [Results](#Results)
* [Visualizing the Results](#Visualizing-the-Results)
* [References](#References)

## Multi Task Learning
We implement a multi-task approach in our network architecture. It helps to gather information from the training signals of related tasks. In our context, it refers to the age and gender estimation. For our experiments, we mainly inherit the Levi-Hassner CNN Model, designed by [1], as the backbone of our model. We implement the multi-task learning approach and add our own head to the backbone which outputs both the age and gender of the image.

![Multi-Task Levi-Hassner Architecture](https://github.com/aneezJaheez/LeviHassner-AgeGender-Prediction/blob/main/img/Architecture.png)

## Overview
This repository contains the model and associated python files to perform age and gender classification on the Adience benchmark dataset. The code is implemented using Python 3.8 and the models are implemented using TensorFlow 2.3. 

The first experiment is the base experiment (which we establish as our benchmark), where we trained the multi-task model on the Adience dataset from scratch and evaluated its accuracy, we will be using this model for basic comparisons. For the second experiment, we pretrained the model using the CelebA dataset before training it with the Adience dataset. For the last experiment, we created two individual models using the Levi-Hassner backbone but each with a head that only outputs the gender or age. These two models were individually trained on the Adience dataset. The results of the last 2 experiments will be compared with the results of the first experiment.

## Dependencies and Environment

The architecture is implemented using TensorFlow 2.3. A full list of dependencies are listed in the environment.yml file. Run the command below to install the required packages in a conda environment. 

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

## Running the Model

After installing the required dependencies and preparing the datasets for training, the training can be run from the project directory of the via the command 

```
python run.py
```

In addition, the attack includes a list of configurations that can be tuned according to the attack requiremenets. A full list of configurations ar can be found in [configs/defaults.py]. The configurations can either be tuned directly in the file, or stated in the initial command to run the attack. For instance, to run model training for both age and gender recognition with a batch size of 50 for 20 epochs using the adam optimizer using a pretrained model and with a runtime seed of 100:

```
python run.py DATA.TARGET_LABEL both DATA.BATCH_SIZE 50 MODEL.OPTIMIZER.NAME adam TRAIN.MAX_EPOCHS 20 MODEL.PRETRAIN_PATH /export/home/aneezahm001/nndl_2/checkpoints/celeba_age_gender RUN_SEED 100
```

The configurations listed on the command line in the above manner take precendence over the configurations in the main configurations file.

## Results

| Classification | Multi-Task Benchmark | Best from [1] | Best from [6] |
| --- | --- | --- | --- |
| Age | 0.480 ± 0.0085 | 0.495 ± 0.44 | 0.451 ± 0.26 |
| Gender | 0.828 ± 0.0054 | 0.859 ± 0.14 | 0.778 ± 0.13 |

Table 1: Our benchmark compared to the best from [1] and [6].


| Classification | Pre-Trained Model | Multi-Task Benchmark |
| --- | --- | --- |
| Age | 0.496 ± 0.012 | 0.480 ± 0.0085 |
| Gender | 0.846 ± 0.0076 | 0.828 ± 0.0054 |

Table 2: Age and Gender results against the benchmark.


| Classification | Multi-Task Benchmark | Single-Task Model |
| --- | --- | --- |
| Age | 0.455 ± 0.0093 | 0.480 ± 0.0085 |
| Gender | 0.814 ± 0.0075 | 0.828 ± 0.0054 |

Table 3: Age and Gender results against the benchmark.

## Visualizing the Results
On every run, the results are logged in the directory "./age_gender/logs". The logs contain the output over the entire training run in the "training.log.tsv" file which can be viewed as a txt file. The logs also contain tensorboard visualizations of the training dataset, validation dataset, and training and validation accuracies and losses over each cross-validation fold in the dataset. These logs can be visualized using tensorboard. For instance, to view the training logs for the task of age and gender recognition, use the following command from the project directory:


```
tensorboard --logdir=./age_gender/logs/adience/1668097217_both
```

Similarly, to view the training visualizations for the task of gender classification:

```
tensorboard --logdir=./age_gender/logs/adience/1668091466_gender
```

## References
[1] G. Levi and T. Hassner, “Age and gender classification using convolutional neural networks”. In IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) workshops, 2015.

[2] S. Ruder. “An Overview of Multi-Task Learning in Deep Neural Networks”. 2017. https://ruder.io/multi-task/index.html

[3] S. Lapuschkin, A. Binder, K. Muller, W. Samek, “Understanding and Comparing Deep Neural Networks for Age and Gender Classification”. Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017, pp. 1629-1638

[4] W.-L. Chao, J.-Z. Liu, and J.-J. Ding. “Facial age estimation based on label-sensitive learning and age-oriented regression”. In Pattern Recognition, 46(3):628–641, 2013.

[5] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. “Backpropagation applied to handwritten zip code recognition”. In Neural computation, 1(4):541–551, 1989.

[6] E. Eidinger, R. Enbar, and T. Hassner. “Age and Gender Estimation of Unfiltered Faces”. 2014. In Transactions on Information Forensics and Security (IEEE-TIFS), special issue on Facial Biometrics in the Wild, 9(12): 2170 - 2179.

[7] Z. Liu, P. Luo, X. Wang, and X. Tang. “Deep learning face attributes in the wild”. 2014. In Proceedings of the IEEE international conference on computer vision (pp. 3730-3738).

[8] M. F. Mustapha, et al. “Age Group Classification using Convolutional Neural Network (CNN)”. 2021. In J. Phys.: Conf. Ser. 2084 012028

[9] T. Hassner, S. Harel, E. Paz, and R. Enbar. “Effective face frontalization in unconstrained images”. Proc. Conf. Comput. Vision Pattern Recognition, 2015. 5, 6

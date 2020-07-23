# CIFAR-10

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

------

## Classes 
<!-- UL -->
* 0: airplane
* 1: automobile
* 2: bird
* 3: cat
* 4: deer
* 5: dog
* 6: frog
* 7: horse
* 8: ship
* 9: truck

--------

## Libraries
<!-- UL -->
* Keras
* pandas
* numpy
* matplotlib
* seaborn
* random

------

## Approach

the dataset is totally balanced. so, after preprocessing the data i used VGG architcture with 3 Blocks with SAME padding. then i tried variations of Dropout Regularization.
my next step is to try Data Augmentation and Batch Normalization.

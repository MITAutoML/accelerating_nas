# Neural Network Performance Prediction Dataset


This repository contains datasets for neural network performance prediction. Each CSV file in the "data" directory contains architecture details, training hyperparamters, and per-epoch validation accuracy for a variety of neural networks trained on image classification and language modeling datasets. Please cite the following paper if you use this dataset:

**[Accelerating Neural Architecture Search using Performance Prediction](https://openreview.net/pdf?id=BJypUGZ0Z)** <br>
*Bowen Baker\*, Otkrist Gupta\*, Ramesh Raskar, Nikhil Naik* <br>
*International Conference on Learning Representations (ICLR) Workshop 2018* <br>
(*\* denotes equal contribution*)

In this paper, we showed that simple regression models can predict the final performance of partially trained model configurations using features based on network architectures, hyperparameters, and partially-trained time-series validation performance data. Using these prediction models, we implemented an early stopping method for hyperparameter optimization and neural architecture search, which obtained a speedup of a factor up to 6x. Our early stopping method can be seamlessly incorporated into both reinforcement learning-based architecture search algorithms and bandit-based hyperparameter search.

We hope that this dataset encourages further research in neural network performance prediction.

## Dataset Description

### 1. MetaQNN CNNs - CIFAR10 (metaqnn\_cifar10.csv) and MetaQNN CNN - SVHN (metaqnn\_svhn.csv)
We sampled CNN architectures from the search space detailed in the [MetaQNN](https://openreview.net/pdf?id=S1c2cvqee) paper by Baker et al. (2017), which allows for varying the numbers and orderings of convolution, pooling, and fully connected layers. The model depths were between 1 and 12 layers for the SVHN experiment and between 1 and 18 layers for the CIFAR-10 experiment. Each model was trained for a total of 20 epochs with the Adam optimizer with &beta;<sub>1</sub> = 0.9, &beta;<sub>2</sub> = 0.999,  and &epsilon;= 10<sup>-8</sup>. The batch size was set to 128. We reduced the learning rate by a factor of 0.2 every 5 epochs. The CSV files contain the following fields:<br>

**net**: This is a string description of the trained neural network. For CNNs, *C(n, f, l)* denotes a Convolutional layer with *n* filters, receptive field size *f*, and stride *l*. *P(f, l)* denotes a Pooling layer with receptive field size *f* and stride *l*. *SM(n)* denotes a *softmax* layer with *n* outputs. *GAP(n)* denotes a Global Average Pooling layer with *n* outputs.<br>

**num\_params**: The total number of parameters in the neural network

**depth**: The total number of layers in the neural network, excluding GAP and SM layers

**lr**: Initial learning rate

**acc\_n**: Validation set accuracy at epoch *n* <br/><br/>

### 2. Deep ResNets - TinyImageNet: (deep\_resnets\_tinyimagenet.csv)
We sampled  ResNet architectures and trained them on the [TinyImageNet](https://tiny-imagenet.herokuapp.com/) dataset (containing 200 classes with 500 training images of 32×32 pixels) for 140 epochs. We varied depths, filter sizes and number of convolutional filter block outputs. Each network was trained for 140 epochs, using Nesterov optimizer. For all models, learning rate reduction and momentum were set to 0.1 and 0.9 respectively. The CSV files contain the following fields:<br>

**num\_blocks**: Number of ResNet blocks in the model. Each ResNet block is composed of three convolutional layers followed by batch normalization and summation layers. We vary the number of blocks from 2 to 18, giving us networks with depths between 14 and 110 (since *depth = 6\*num\_blocks + 2*)

**num\_filters**: Number of filters in each convolutional layer. Sampled from {2, 3, ..., 22}

**filter_size**: Filter size for each convolutional layer. Sampled from {3, 5, 7}

**lr**: Initial learning rate

**ss**: Learning rate reduction step size

**acc\_n**: Validation set accuracy at epoch *n*. Note that *acc_0* refers to the accuracy of a randomly initialized model. <br/><br/>



### 3. Deep ResNets - CIFAR10: (deep\_resnets\_cifar10.csv) 
We sampled ResNet architectures from a search space similar to [Zoph & Le (2017)](https://openreview.net/pdf?id=r1Ue8Hcxg). Each architecture consists of 39 layers: 12 *conv*, a 2x2 *max pool*, 9 *conv*, a 2x2 *max pool*, 15 *conv*, and *softmax*. Each *conv* layer is followed by batch normalization and *ReLU*. A ResNet block contains 3 *conv* layers, which share the same kernel width, kernel height, and number of learnable kernels. We varied these parameters across architectures. Each network was trained on the CIFAR-10 dataset for 50 epochs using the RMSProp optimizer, with
variable initial learning rate, weight decay of 10<sup>−4</sup>, and a learning rate reduction to 10<sup>−5</sup> at epoch 30. The CSV files contain the following fields:<br>

**num\_params**: The total number of parameters in the neural network.

**lr**: Initial learning rate

**acc\_n**: Validation set accuracy at epoch *n*. Note that *acc_0* refers to the accuracy of a randomly initialized model. <br/><br/>



### 4. CudaConvnet - CIFAR10 (cudaconvnet\_cifar10.csv) and CudaConvet - SVHN (metaqnn\_svhn.csv)
We trained the [Cuda-Convnet](https://code.google.com/p/cuda-convnet/) architecture with varying values of initial learning rate, learning rate reduction step size, weight decay for convolutional and fully connected layers, and scale and power of local response normalization layers. We trained models with CIFAR-10 for 60 epochs and with SVHN for 12 epochs.  The CSV files contain the following fields:<br>

**lr**: Initial learning rate

**ss**: Learning rate reduction step size

**cn_l2**: L_2 penalty of Conv"n" layer

**f4_l2**: L_2 penalty for FC4  layer

**weight_decay**: Weight decay parameter     

**lrn_scale**: Response normalization scale

**lrn_pow**: Response normalization power

**acc\_n**: Validation set accuracy at epoch *n*. <br/><br/>



### 5. LSTM - Penn Treebank (lstm\_ptb.csv)
We sampled LSTM models and trained them on the Penn Treebank dataset for 60 epochs. Both the number of hidden layer inputs and number of LSTM cells were varied between 10 to 1400 in steps of 20. Each network was trained with SGD for 60 epochs with batch size of 50 and dropout ratio of 0.5. We used a dictionary size of 400 words to generate embeddings when vectorizing the data. The CSV files contain the following fields:<br>

**n_inputs**: number of LSTM cells

**n_layers**: depth of neural network

**base_lr**: Initial learning rate

**ss**: Learning rate reduction step size

**acc\_n**: Validation set perplexity at epoch *n*. Note that *acc_0* refers to the perplexity of a randomly initialized model.<br/><br/>



### 6. AlexNet - ImageNet (alexnet\_imagenet.csv) 

We trained the AlexNet model on the original ImageNet (ILSVRC12) dataset. To
compensate for our limited computation resources, we randomly sampled 10% of dataset, trained each
configuration for 500 epochs, and only varied initial learning rate and learning rate reduction step size across models.

**lr**: Initial learning rate

**ss**: Learning rate reduction step size

**acc\_n**: Validation set accuracy at epoch *n*. Note that *acc_0* refers to the accuracy of a randomly initialized model.<br/><br/>



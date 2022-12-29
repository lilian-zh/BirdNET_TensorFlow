####################
'''
These codes are adapted based on the scripts provided by Lea. We updated some model and functions to fit our own targets.
'''
####################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings


import tensorflow as tf
import tensorflow.keras.backend as K

layers = tf.keras.layers

import params as params


##############################################
'''
buildNet is translated from BirdNET(https://github.com/kahst/BirdNET) using tensorflow.
'''
NONLINEARITY = 'relu'
FILTERS = [8, 16, 32, 64, 128]
KERNEL_SIZES = [(5, 5), (3, 3), (3, 3), (3, 3), (3, 3)]
RESNET_K = 4
RESNET_N = 3



def resblock(net_in, filters, kernel_size, filter_connect, stride=1, preactivated=True, block_id=1, name=''):
    # Pre-activation
    if block_id > 1:
        net_pre = layers.Activation('relu')(net_in)
    else:
        net_pre = net_in
    
    # Pre-activated shortcut?
    if preactivated:
        net_in = net_pre

    # Bottleneck Convolution
    if stride > 1: 
        net_pre = layers.Conv2D(filter_connect, (1, 1), strides=(1, 1), activation='relu', padding='same')(net_pre)   
        net_pre = layers.BatchNormalization()(net_pre)
    
    ## First Convolution 
    net = layers.Conv2D(filter_connect, kernel_size, strides=(1, 1), activation='relu', padding='same')(net_pre)   
    net = layers.BatchNormalization()(net)
    # Pooling layer
    if stride > 1:
        net = layers.MaxPooling2D((stride, stride), padding='valid')(net)
    # Dropout Layer
    net = layers.Dropout(0.5)(net)

    ## Second Convolution     
    net = layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='same')(net)   
    net = layers.BatchNormalization()(net)

    # Shortcut Layer
    if stride > 1:

        # Average pooling
        shortcut = layers.AveragePooling2D(pool_size=(stride, stride), strides=stride, padding='valid')(net_in)

        # Shortcut convolution
        shortcut = layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(shortcut)   
        shortcut = layers.BatchNormalization()(shortcut)    
        
    else:

        # Shortcut = input
        shortcut = net_in
    
    # Merge Layer
    out = layers.Add()([net, shortcut])


    return out


def classificationBranch(net, kernel_size):

    # Post Convolution
    branch = layers.Conv2D(int(FILTERS[-1] * RESNET_K), kernel_size, strides=(1, 1), activation='relu', padding='valid')(net)   
    branch = layers.BatchNormalization()(branch)

    # Dropout Layer
    branch = layers.Dropout(0.5)(branch)
    
    # Dense Convolution
    branch = layers.Conv2D(int(FILTERS[-1] * RESNET_K * 2), (1, 1), strides=(1, 1), activation='relu', padding='valid')(branch)   
    branch = layers.BatchNormalization()(branch)
  
    # Dropout Layer
    branch = layers.Dropout(0.5)(branch)
    
    # Class Convolution
    branch = layers.Conv2D(params.NB_CLASSES, (1, 1), strides=(1, 1), padding='valid')(branch)

    return branch



def buildNet(input_shape=(params.HEIGHT, params.WIDTH, 1),pooling='avg'):
    # Input layer for images
    aud_input = layers.Input(shape=input_shape)

    # Pre-processing stage
    x = layers.Conv2D(int(FILTERS[0] * RESNET_K), (5, 5), strides=(1, 1), activation='relu', padding='same')(aud_input)   
    x = layers.BatchNormalization()(x)

    # Max pooling
    x = layers.MaxPooling2D((1, 2), strides=(1, 2), padding='valid')(x)

    # Residual Stacks
    filter_connect = int(FILTERS[0] * RESNET_K)
    for i in range(1, len(FILTERS)):
        x = resblock(x,
                    filters=int(FILTERS[i] * RESNET_K),
                    kernel_size=KERNEL_SIZES[i],
                    filter_connect=filter_connect,
                    stride=2,
                    preactivated=True,
                    block_id=i,
                    name='BLOCK ' + str(i) + '-1')
        filter_connect = int(FILTERS[i] * RESNET_K)
        for j in range(1, RESNET_N):
            x = resblock(x,
                        filters=int(FILTERS[i] * RESNET_K),
                        kernel_size=KERNEL_SIZES[i],
                        filter_connect=filter_connect,
                        preactivated=False,
                        block_id=i+j,
                        name='BLOCK ' + str(i) + '-' + str(j + 1))

    
    # Post Activation
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Classification branch
    x = classificationBranch(x,  (4, 10))


    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D()(x)
    elif pooling == 'logsumexp':
        # x = tf.math.reduce_logsumexp(x, axis=None, keepdims=False, name=None)
        # need layer, instead of function
        pass
    elif pooling == 'logmeanexp':
        pass
    else:
        warnings.warn('Please choose correct pooling for "buildNet(include_top=False)"')


    inputs = aud_input

    # Create model.
    model = tf.keras.Model(inputs, x, name='buildNet')


    return model

  
        
##############################################
def resnet50():
    model = tf.keras.applications. ResNet50(
        include_top=False, weights=None, input_tensor=None, input_shape=(params.HEIGHT, params.WIDTH, 3), #should have 3 channels
        pooling='avg', classes=params.NB_CLASSES
    )
    return model

 ###############################################  
# Dictionary of all the models defined here
MODELS = {
     'buildNet': buildNet(),
     'resnet': resnet50()
    }

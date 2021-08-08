# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 12:04:52 2021

@author: Abdelrahman
"""

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import concatenate
from keras import Model

class CIFAR10Net(object):
    
    def __inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
        #1x1 conv
        conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
        
        #3x3 conv
        conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)
        conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)
        
        #5x5 conv
        conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)
        conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)
        
        #3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
        pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)
        
        #concatenate filters, assumes filters/channels last
        layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
        
        return layer_out


    def build(self, input_shape: tuple, n_blocks: int, architecture: dict):
        '''
        Parameters
        ----------
        input_shape : tuple
            DESCRIPTION. --> The shape of the input layer
        n_blocks : int
            DESCRIPTION. --> Number of inception module blocks needs to be generated. 
        architecture : dict
            DESCRIPTION. --> e.g. {'block_1': [64, 96, 128, 16, 32, 32],
                                   'block_2': [128, 128, 192, 32, 96, 64]}
        
        Returns
        -------
        keras Model.

        '''
        
        input_ = Input(shape = input_shape)
        
        f1, f2_in, f2_out, f3_in, f3_out, f4_out = architecture[list(architecture)[0]]
        block = self.__inception_module(input_, f1, f2_in, f2_out, f3_in, f3_out, f4_out)
        
        for idx in range(1, len(architecture.keys())):
            f1, f2_in, f2_out, f3_in, f3_out, f4_out = architecture[list(architecture)[idx]]
            block = self.__inception_module(block, f1, f2_in, f2_out, f3_in, f3_out, f4_out)
            
        
        flat = Flatten()(block)
        hidden = Dense(1024, activation='relu')(flat)
        drop = Dropout(0.5)(hidden)
        output = Dense(10, activation='softmax')(drop)
        
        
        model = Model(inputs = [input_], outputs = [output])
        
        return model
            
            
        

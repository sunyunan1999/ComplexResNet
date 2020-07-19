import numpy as np
import argparse
from keras.layers import Input, Activation, Subtract
from keras.models import Model
#from ..deep_complex_networks-master.complexnn import ComplexConv2D, ComplexBN, ComplexDense
from conv import ComplexConv2D
from bn_v2 import ComplexBN
from dense import ComplexDense
parse = argparse.ArgumentParser(description="Processing configurations ...")
parse.add_argument('--model', default='DimResNN', type=str, help='Choose a model')
parse.add_argument('--train_data', default='./train_data/', type=str, help='the directory of train data')
parse.add_argument('--batch_size', default=120, type=int, help='batch size for training')
parse.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate for training')
parse.add_argument('--epochs', default=50, type=int, help='The number of epochs')


def DimResNN(depth, filters, kernal_size,use_bn=True):  #input kernalsize syn
    layer_count = 0
  #  inputs = Input(shape=(None,None,1), name='Input' + str(layer_count))  #syn
    inputs = Input(shape=(None,None,6), name='Input' + str(layer_count))  #syn 这个设置应该有问题，原本设置是NONE,NOEN,1
    outs = inputs
    layer_count += 1

    # the first two parameters of ComplexConv2D are the number of filters and kernal size.
    # They are set based on a specific design
    outs = (ComplexConv2D(filters, kernal_size, strides=(1,1), padding='same',
                          activation='linear', kernel_initializer='complex_independent',
                          name='ComplexConv'+str(layer_count)))(outs)
    layer_count += 1
    outs = (Activation('relu', name='relu'+str(layer_count)))(outs)      # In this way, it implements 
                                                                         # the CReLU activation function

    # depth-2 layers of ComplexConv2 + ComplexBN + RELU
    for i in range(depth-2):
        layer_count += 1
        outs = (ComplexConv2D(filters, kernal_size, strides=(1,1), padding='same',
                             activation='linear', kernel_initializer='complex_independent',
                             name='ComplexConv' + str(layer_count)))(outs)

        if use_bn:
            layer_count += 1
 #            outs = (ComplexBN(axis=-1, name='ComplexBN'+str(layer_count)))(outs)   #syn 需要另外8个参数,否则无法归一化

        layer_count += 1
        outs = (Activation('relu', name='relu'+str(layer_count)))(outs)
    # last conv layer
    layer_count += 1
    outs = (ComplexConv2D(filters, kernal_size, strides=(1,1), padding='same',
                             activation='linear', kernel_initializer='complex_independent',
                             name='ComplexConv'+str(layer_count)))(outs)
    layer_count += 1
    outs = Subtract(name='subtract'+str(layer_count))([inputs,outs])
    model = Model(inputs=inputs, outputs=outs)

    return model











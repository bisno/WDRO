#-*- coding: UTF-8 -*- 
# Author xuli.shen

import tensorflow as tf


def new_weights(name,shape):

    return tf.get_variable(name ,shape, initializer=tf.contrib.layers.xavier_initializer())


def new_biases(length):
    return tf.Variable(tf.constant(0.01, shape=[length]))

def new_fc_layer(name,
                 input,          
                 num_inputs,     
                 num_outputs,    
                 use_relu=True): 

    weights = new_weights(name,shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer, weights



def new_conv_layer(name,
                    input,              
                   num_input_channels, 
                   filter_size,        
                   num_filters,       
                   use_pooling=True):  


    shape = [filter_size, filter_size, num_input_channels, num_filters]

    weights = new_weights(name,shape=shape)

    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    if use_pooling:

        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    layer = tf.nn.relu(layer)

    return layer, weights

def flatten_layer(layer):

    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

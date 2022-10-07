# From https://github.com/taki0112/ResNet-Tensorflow.

import time
from ResNets.ops import *

import glob
import numpy as np
import pandas as pd
from PIL import Image
import random
import time


def network(x, res_n=18, is_training=True, reuse=False):
    with tf.variable_scope("network", reuse=reuse):

        if res_n < 50:
            residual_block = resblock
        else:
            residual_block = bottle_resblock

        residual_list = get_residual_layer(res_n)

        ch = 32
        x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

        for i in range(residual_list[0]):
            x = residual_block(x, channels=ch, is_training=is_training,
                               downsample=False, scope='resblock0_' + str(i))

        x = residual_block(x, channels=ch*2, is_training=is_training,
                           downsample=True, scope='resblock1_0')

        for i in range(1, residual_list[1]):
            x = residual_block(x, channels=ch*2, is_training=is_training,
                               downsample=False, scope='resblock1_' + str(i))

        x = residual_block(x, channels=ch*4, is_training=is_training,
                           downsample=True, scope='resblock2_0')

        for i in range(1, residual_list[2]):
            x = residual_block(x, channels=ch*4, is_training=is_training,
                               downsample=False, scope='resblock2_' + str(i))

        x = residual_block(x, channels=ch*8, is_training=is_training,
                           downsample=True, scope='resblock_3_0')

        for i in range(1, residual_list[3]):
            x = residual_block(x, channels=ch*8, is_training=is_training,
                               downsample=False, scope='resblock_3_' + str(i))

        x = batch_norm(x, is_training, scope='batch_norm')
        x = relu(x)

        x = fully_conneted(x, units=5120, scope='logit')

        return x

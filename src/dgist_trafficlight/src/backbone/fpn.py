#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from DEFINE import *
import tensorflow.contrib.slim as slim
from collections import OrderedDict
conv2d = H['conv2d']

import rospy

def upsample_nn(x, ratio):
    if H['data_format'] == 'NCHW':
        x = tf.transpose(x, [0, 2, 3, 1])

    s = tf.shape(x)
    h = s[1]
    w = s[2]
    rt = tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])
    if H['data_format'] == 'NCHW':
        rt = tf.transpose(rt, [0, 3, 1, 2])
    return rt


def top_down_op(top, down, depth=256, scope=None):
    top_down = top
    top_down = upsample_nn(top_down, 2)
    residual = conv2d(down, depth, 1, activation_fn=None, normalizer_fn=None, scope=scope)
    top_down += residual
    return top_down


class FPN(object):

    def __init__(self, features):
        self.fpn_feature = self.build_fpn(features)

    def build_fpn(self, features):
        batch_norm_params = {
            'is_training': False,
            'decay': H['batch_norm_decay'],
            'epsilon': H['batch_norm_epsilon'],
            'scale': H['batch_norm_scale'],
            'updates_collections': H['updates_collections'],
            'fused': H['fused'],
            'data_format': H['data_format']
        }

        with tf.variable_scope('decoder'):
            with slim.arg_scope(
                    [conv2d],
                    weights_initializer=slim.variance_scaling_initializer(),
                    activation_fn=activation_fn,
                    normalizer_fn=normalization,
                    normalizer_params={'is_training': False}):

                depth = 256
                with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                    M5 = conv2d(features['C5'], depth, 1, activation_fn=None, normalizer_fn=None, scope='M5')
                    M4 = top_down_op(M5, features['C4'], scope='M4')
                    M3 = top_down_op(M4, features['C3'], scope='M3')

                    P5 = conv2d(M5, depth, 3, activation_fn=None, normalizer_fn=None, scope='P5')
                    P4 = conv2d(M4, depth, 3, activation_fn=None, normalizer_fn=None, scope='P4')
                    P3 = conv2d(M3, depth, 3, activation_fn=None, normalizer_fn=None, scope='P3')

                    P6 = conv2d(features['C6'], depth, 3, activation_fn=None, normalizer_fn=None, scope='P6')
                    P7 = conv2d(features['C7'], depth, 3, activation_fn=None, normalizer_fn=None, scope='P7')

                    P3 = slim.batch_norm(P3)
                    P4 = slim.batch_norm(P4)
                    P5 = slim.batch_norm(P5)
                    P6 = slim.batch_norm(P6)
                    P7 = slim.batch_norm(P7)

                    P4 = conv2d(P3, depth, 3, 2, scope='P3-up', normalizer_fn=None) + P4
                    P5 = conv2d(P4, depth, 3, 2, scope='P4-up', normalizer_fn=None) + P5
                    P6 = conv2d(P5, depth, 3, 2, scope='P5-up', normalizer_fn=None) + P6
                    P7 = conv2d(P6, depth, 3, 2, scope='P6-up', normalizer_fn=None) + P7

        fpn_feature = OrderedDict()
        fpn_feature['P3'] = P3
        fpn_feature['P4'] = P4
        fpn_feature['P5'] = P5
        fpn_feature['P6'] = P6
        fpn_feature['P7'] = P7

        return fpn_feature

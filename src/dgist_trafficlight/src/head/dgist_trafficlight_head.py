# -*- coding: utf-8 -*-

"""
Brief  :
작성자 : 박재형(21.10.)
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from collections import OrderedDict
from DEFINE import *


conv2d = H['conv2d']


class DgistTrafficlightHead(object):
    def __init__(self, features):
        self.out = self.build_trafficlight_head(features)

    def build_output(self, hidden_class, stride, scope=''):
        with tf.variable_scope(scope):
            shape = tf.shape(hidden_class)
            flatten_class = tf.reshape(hidden_class, [shape[0], int(256 / stride * 128 / stride * 256)])
            pred_confidences = tf.layers.dense(inputs=flatten_class, units=6, activation=tf.nn.relu)

        return {'pred_confidences': pred_confidences}

    def build_trafficlight_head(self, features):
        FPN = [features['P3'], features['P4'], features['P5'], features['P6'], features['P7']]

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
                    out = OrderedDict()
                    for idx, p in enumerate(FPN):
                        with slim.arg_scope([conv2d], weights_initializer=tf.random_normal_initializer(stddev=0.01), deformable=False):
                            with tf.variable_scope('head', reuse=tf.AUTO_REUSE):
                                hidden_class = conv2d(p, depth, 3, normalizer_fn=None, scope='conv1')
                                hidden_class = conv2d(hidden_class, depth, 3, normalizer_fn=None, scope='conv2')
                                hidden_class = conv2d(hidden_class, depth, 3, normalizer_fn=None, scope='conv3')
                                hidden_class = conv2d(hidden_class, depth, 3, activation_fn=None, normalizer_fn=None, scope='conv4')

                                with tf.variable_scope('class_BN%d' % idx, reuse=tf.AUTO_REUSE):
                                    hidden_class = tf.nn.relu(slim.batch_norm(hidden_class))

                                stride = 2 ** idx * 8
                                out['%d' % stride] = self.build_output(hidden_class, stride=stride, scope='dgist_%d' % stride)

        keys = [key for key in out]
        concat_class = tf.concat([out[key]['pred_confidences'] for key in keys], 1)
        final_result = {'pred_class': tf.layers.dense(inputs=concat_class, units=6, activation=tf.nn.softmax)}
        return final_result

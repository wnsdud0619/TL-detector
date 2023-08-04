#!/usr/bin/env python

from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time

import rospy

H = {}
H['back_bone'] = 'resnet50'
H['batch_norm_decay'] = 0.99
H['batch_norm_epsilon'] = 1e-5
H['batch_norm_scale'] = True
H['updates_collections'] = None
H['fused'] = True
H['data_format'] = 'NHWC'
H['dtype'] = tf.float32

def activation_fn(input):
    output = tf.nn.relu(input)
    return output

def load_ckpt(sess, variables, path):
    from tensorflow.python import pywrap_tensorflow

    reader = pywrap_tensorflow.NewCheckpointReader(path)
    map = reader.get_variable_to_shape_map()
    def name_in_checkpoint(var):
        name = var.op.name
        if name in map:
            if var.shape != map[name]:
                return False
            print(name + ' exists')
            return name
        else:
            print(name + ' not exist in ' + path)
            return False

    variables_to_restore = {name_in_checkpoint(var): var for var in variables if name_in_checkpoint(var) is not False}
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, path)


def normalization(input, is_training, epsilon=1e-5):
    from DEFINE import H
    batch_norm_params = {
        'is_training': is_training,
        'decay': 0.999,
        'epsilon': H['batch_norm_epsilon'],
        'scale': H['batch_norm_scale'],
        'updates_collections': H['updates_collections'],
        'fused': H['fused'],
        'data_format': H['data_format']
    }

    output = slim.batch_norm(input, **batch_norm_params)
    return output


@slim.add_arg_scope
def _conv2d(inputs, num_outputs, kernel_size,
            stride=1, rate=1, scope=None,
            activation_fn=tf.nn.sigmoid,
            normalizer_fn=None, normalizer_params=None,
            data_format=H['data_format'],
            padding='SAME',
            weights_initializer=None,
            deformable=False,
            keep_prob=None):

    if data_format == 'NHWC':
        in_channel = inputs.get_shape().as_list()[-1]
    else:
        in_channel = inputs.get_shape().as_list()[1]

    if type(kernel_size) is not list:
        kernel_size = [kernel_size, kernel_size]

    with tf.variable_scope(scope):
        filter = tf.get_variable('weights', kernel_size + [in_channel, num_outputs], dtype=H['dtype'], initializer=weights_initializer, trainable=True)
        if stride > 1:
            padding = 'VALID'
            if kernel_size[0] > 2 and (not deformable or stride != 1):
                kernel_size_effective = kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)
                pad_total = kernel_size_effective - 1
                pad_beg = pad_total // 2
                pad_end = pad_total - pad_beg

                if data_format == 'NHWC':
                    inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
                else:
                    inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])

        if data_format == 'NHWC':
            output = tf.nn.conv2d(inputs, filter, [1, stride, stride, 1], padding, dilations=[1, rate, rate, 1], name=scope, data_format=data_format)
        else:
            output = tf.nn.conv2d(inputs, filter, [1, 1, stride, stride], padding, dilations=[1, 1, rate, rate], name=scope, data_format=data_format)

        tf.add_to_collection('checkpoints', output)

        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            output = normalizer_fn(output, **normalizer_params)
        else:
            bias = tf.get_variable('biases', [num_outputs], dtype=H['dtype'], initializer=tf.zeros_initializer, trainable=True)
            output = tf.nn.bias_add(output, bias, data_format=data_format)

        print(activation_fn)
        print(tf.get_variable_scope().name)

        if activation_fn is not None:
            output = activation_fn(output)

            if keep_prob is not None:
                output = tf.nn.dropout(output, keep_prob=keep_prob)
    return output

H['conv2d'] = _conv2d

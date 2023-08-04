#!/usr/bin/env python

import tensorflow as tf
from DEFINE import *
import tensorflow.contrib.slim as slim

MEAN_VALUE = [103.062623801, 115.902882574, 123.151630838]
conv2d = H['conv2d']

import rospy

class resnet(object):
    def __init__(self, phase_handler, depth=101):
        self._train = phase_handler
        self.train = False
        self.depth = depth
        if self.depth == 50:
            self.num_block3 = 6
            self.scope = 'resnet_v1_50'

    def build_model(self, input_tensor):
        out = {}
        red, green, blue = tf.split(input_tensor, 3, axis=3)
        bgr = tf.concat([blue, green, red], 3) - MEAN_VALUE
        if H['data_format'] == 'NCHW':
            bgr = tf.transpose(bgr, [0, 3, 1, 2])

        batch_norm_params = {
            'is_training': self.train,
            'decay': H['batch_norm_decay'],
            'epsilon': H['batch_norm_epsilon'],
            'scale': H['batch_norm_scale'],
            'updates_collections': H['updates_collections'],
            'fused': H['fused'],
        }

        with tf.variable_scope(self.scope, 'resnet_v1'):
            with slim.arg_scope(
                    [conv2d],
                    weights_initializer=slim.variance_scaling_initializer(),
                    activation_fn=activation_fn,
                    normalizer_fn=normalization,
                    deformable=False,
                    normalizer_params={'is_training': self.train}):

                with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                    with slim.arg_scope([slim.max_pool2d], padding='SAME'):
                        net = conv2d(bgr, 64, 7, stride=2, scope='conv1')
                        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', data_format=H['data_format'])

                        conv2_end = self.residual_block('block1', net, 3, {'depth_out': 256, 'depth_bottleneck': 64}, block_stride=1)
                        self.conv3_end = self.residual_block('block2', conv2_end, 4, {'depth_out': 512, 'depth_bottleneck': 128}, block_stride=2)
                        self.conv4_end = self.residual_block('block3', self.conv3_end, self.num_block3, {'depth_out': 1024, 'depth_bottleneck': 256}, block_stride=2)
                        self.conv5_end = self.residual_block('block4', self.conv4_end, 3, {'depth_out': 2048, 'depth_bottleneck': 512}, block_stride=2)
                        self.conv6_end = self.residual_block('block5', self.conv5_end, 3, {'depth_out': 1024, 'depth_bottleneck': 256}, block_stride=2)
                        self.conv7_end = self.residual_block('block6', self.conv6_end, 3,  {'depth_out': 1024, 'depth_bottleneck': 256}, block_stride=2)

        out['C2'] = conv2_end
        out['C3'] = self.conv3_end
        out['C4'] = self.conv4_end
        out['C5'] = self.conv5_end
        out['C6'] = self.conv6_end
        out['C7'] = self.conv7_end
        return out

    def residual_block(self, name, input_tensor, num_unit, depth, block_stride=2):
        _input = input_tensor
        with tf.variable_scope(name):
            for i in range(num_unit):
                with tf.variable_scope('unit_%d' % (i+1)+'/bottleneck_v1'):
                    depth_out = depth['depth_out']
                    depth_bottleneck = depth['depth_bottleneck']
                    if i != 0:
                        short_cut = self.subsample(_input, 1, 'shortcut')
                        residual = conv2d(_input, depth_bottleneck, [1, 1], scope='conv1')
                    else:
                        short_cut = conv2d(_input, depth_out, [1, 1], block_stride, activation_fn=None, scope='shortcut')
                        residual = conv2d(_input, depth_bottleneck, [1, 1], block_stride, scope='conv1')

                    keep_prob = None
                    if self._train:
                        keep_prob = 0.8
                    residual = conv2d(residual, depth_bottleneck, 3, scope='conv2', keep_prob=keep_prob)
                    residual = conv2d(residual, depth_out, [1, 1], activation_fn=None, scope='conv3')
                    _input = tf.nn.relu(short_cut + residual)
        return _input

    def subsample(self, inputs, factor, scope=None):
        if factor == 1:
            return inputs
        else:
            return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

        def name_in_checkpoint(var):
            if self.prefix in var.op.name:
                name = var.op.name.replace(self.prefix + '/' + self.scope, self.scope)
                if name in map:
                    print(name + ' exists')
                    return name
                else:
                    print(name + ' not exist in ' + H['resnet50_path'])
                    return False
            else:
                print('prefix not found')
                print('op_name:', var.op.name)
                return False

        variables_to_restore = {name_in_checkpoint(var): var for var in variables if name_in_checkpoint(var) is not False}
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, H['resnet50_path'])
        print('resnet loaded')

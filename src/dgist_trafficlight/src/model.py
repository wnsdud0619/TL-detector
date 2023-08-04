#!/usr/bin/env python

from __future__ import print_function
from DEFINE import *
from backbone import resnet
from backbone.fpn import FPN
from head.dgist_trafficlight_head import DgistTrafficlightHead


class MultinetSeed(object):
    def __init__(self, sess, image):
        self.sess = sess
        self.input = image
        self.reuse_variables = False
        self.out = {}
        self.scope_name = 'model'

        with tf.variable_scope(self.scope_name, reuse=self.reuse_variables):
            self.build_backbone()
            self.out = self.build_head()

    def build_backbone(self):
        with tf.variable_scope('encoder', reuse=self.reuse_variables):
            self.resnet = resnet.resnet(False, 50)
            self.features = self.resnet.build_model(self.input)

        with tf.variable_scope('detection', reuse=self.reuse_variables):
            fpn = FPN(self.features)
            self.features.update(fpn.fpn_feature)

    def build_head(self):
        with tf.variable_scope('dgist_tl', reuse=self.reuse_variables):
            model_out = DgistTrafficlightHead(self.features)
        return model_out.out

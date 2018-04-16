import numpy as np
import scipy.io
import scipy.misc
import argparse
import sys
import tensorflow as tf

CONTENT_LAYERS = [('conv4_2', 1.)]
STYLE_LAYERS = [
    ('conv1_1', 1.),
    ('conv2_1', 1.),
    ('conv3_1', 1.),
    ('conv4_1', 1.),
    ('conv5_1', 1.)]

MEANS = np.array([123, 117, 104]).reshape((1, 1, 1, 3))
VGG_MODEL='imagenet-vgg-verydeep-19.mat'

def _conv_layer(input, w_b=None):
    return tf.nn.relu(tf.nn.conv2d(input, w_b[0], strides=[1, 1, 1, 1], padding='SAME') + w_b[1])


def _avg_pool(input):
    return tf.nn.avg_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def _weight(vgg_layers, index):
    #pdb.set_trace()
    w = vgg_layers[index][0][0][0][0][0]
    b = vgg_layers[index][0][0][0][0][1]
    w = tf.constant(w)
    b = tf.constant(np.reshape(b, (b.size)))
    return w, b


def load_vgg_model(path, hight, width):
    model = {}
    vgg_layers = scipy.io.loadmat(path)['layers'][0]
    model['input'] = tf.Variable(np.zeros((1, hight, width, 3)).astype('float32'))
    model['conv1_1'] = _conv_layer(model['input'], _weight(vgg_layers, 0))
    model['conv1_2'] = _conv_layer(model['conv1_1'], _weight(vgg_layers, 2))
    model['pool1'] = _avg_pool(model['conv1_2'])
    model['conv2_1'] = _conv_layer(model['pool1'], _weight(vgg_layers, 5))
    model['conv2_2'] = _conv_layer(model['conv2_1'], _weight(vgg_layers, 7))
    model['pool2'] = _avg_pool(model['conv2_2'])
    model['conv3_1'] = _conv_layer(model['pool2'], _weight(vgg_layers, 10))
    model['conv3_2'] = _conv_layer(model['conv3_1'], _weight(vgg_layers, 12))
    model['conv3_3'] = _conv_layer(model['conv3_2'], _weight(vgg_layers, 14))
    model['conv3_4'] = _conv_layer(model['conv3_3'], _weight(vgg_layers, 16))
    model['pool3'] = _avg_pool(model['conv3_4'])
    model['conv4_1'] = _conv_layer(model['pool3'], _weight(vgg_layers, 19))
    model['conv4_2'] = _conv_layer(model['conv4_1'], _weight(vgg_layers, 21))
    model['conv4_3'] = _conv_layer(model['conv4_2'], _weight(vgg_layers, 23))
    model['conv4_4'] = _conv_layer(model['conv4_3'], _weight(vgg_layers, 25))
    model['pool4'] = _avg_pool(model['conv4_4'])
    model['conv5_1'] = _conv_layer(model['pool4'], _weight(vgg_layers, 28))
    model['conv5_2'] = _conv_layer(model['conv5_1'], _weight(vgg_layers, 30))
    model['conv5_3'] = _conv_layer(model['conv5_2'], _weight(vgg_layers, 32))
    model['conv5_4'] = _conv_layer(model['conv5_3'], _weight(vgg_layers, 34))
    model['pool5'] = _avg_pool(model['conv5_4'])
    return model


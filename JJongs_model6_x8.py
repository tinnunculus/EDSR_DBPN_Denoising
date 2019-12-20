import tensorflow as tf
import numpy as np
import os, glob, time, cv2, math
import matplotlib.pyplot as plt

n_train = 27958
n_val = 3598


class JJONGS_MODEL6_x8(object):
    def __init__(self, sess, input_size = 32, label_size = 256, batch_size = 16, pretrain = False, scale = 8, epoch = 30,  checkpoint_dir = "jjongs_tensorflow_checkpoint", merge = False) :

        self.sess = sess
        self.input_size = input_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.scale = scale
        self.epoch = epoch
        self.checkpoint_dir = checkpoint_dir
        self.merge = merge
        self.build_model()

    def residual_block(self, x, weight1, weight2, bias1, bias2):
        y = tf.nn.conv2d(x, weight1, strides = [1,1,1,1], padding ='SAME') + bias1
        y = tf.nn.relu(y)
        y = tf.nn.conv2d(y, weight2, strides = [1,1,1,1], padding = 'SAME') + bias2
        y = tf.nn.relu(y)
        y = y * 0.3
        return x + y

    def backprojection_upsample(self, x, weight1, weight2, weight3, bias1, bias2, bias3):
        temp = tf.nn.conv2d(x, weight1, strides = [1,1,1,1], padding = 'SAME') + bias1
        temp = tf.nn.relu(temp)
        temp = self.shuffle_operator(temp,8)
        y = tf.nn.conv2d(temp, weight2, strides = [1,8,8,1], padding = 'SAME') + bias2
        y = tf.nn.relu(y)
        z = y - x
        k = tf.nn.conv2d(z, weight3, strides = [1,1,1,1], padding = 'SAME') + bias3
        k = tf.nn.relu(k)
        k = self.shuffle_operator(k,8)
        k = k * 0.1
        return k + temp

    def backprojection_downsample(self, x, weight1, weight2, weight3, bias1, bias2, bias3):
        temp = tf.nn.conv2d(x, weight1, strides = [1,8,8,1], padding = 'SAME') + bias1
        temp = tf.nn.relu(temp)
        y = tf.nn.conv2d(temp, weight2, strides = [1,1,1,1], padding = 'SAME') + bias2
        y = tf.nn.relu(y)
        y = self.shuffle_operator(y,8)
        z = y - x
        k = tf.nn.conv2d(z, weight3, strides = [1,8,8,1], padding = 'SAME') + bias3
        k = tf.nn.relu(k)
        k = k * 0.1
        return k + temp

    def shuffle_operator(self, X, r):
        temp = tf.depth_to_space(X, r)
        return temp

    def build_model(self):
        self.input_images = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3], name = 'input_images')
        self.label_images = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, 3], name = 'output_images')
        self.input_batch_size = tf.placeholder(tf.float32, None)

        with tf.name_scope("parameters"):
            self.weights = {
                'init_conv_weight' : tf.Variable(tf.random_normal([3, 3, 3, 256], stddev=1e-2), name='Init_Conv_Weight'),
                'ups1_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups1_1_Weight'),
                'ups1_2_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Ups1_2_Weight'),
                'ups1_3_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups1_3_Weight'),
                'downs1_1_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs1_1_Weight'),
                'downs1_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Downs1_2_Weight'),
                'downs1_3_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs1_3_Weight'),
                'ups2_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups2_1_Weight'),
                'ups2_2_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Ups2_2_Weight'),
                'ups2_3_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups2_3_Weight'),
                'downs2_1_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs2_1_Weight'),
                'downs2_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Downs2_2_Weight'),
                'downs2_3_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs2_3_Weight'),
                'ups3_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups3_1_Weight'),
                'ups3_2_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Ups3_2_Weight'),
                'ups3_3_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups3_3_Weight'),
                'downs3_1_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs3_1_Weight'),
                'downs3_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Downs3_2_Weight'),
                'downs3_3_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs3_3_Weight'),
                'ups4_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups4_1_Weight'),
                'ups4_2_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Ups4_2_Weight'),
                'ups4_3_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups4_3_Weight'),
                'downs4_1_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs4_1_Weight'),
                'downs4_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Downs4_2_Weight'),
                'downs4_3_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs4_3_Weight'),
                'ups5_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups5_1_Weight'),
                'ups5_2_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Ups5_2_Weight'),
                'ups5_3_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups5_3_Weight'),
                'downs5_1_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs5_1_Weight'),
                'downs5_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Downs5_2_Weight'),
                'downs5_3_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs5_3_Weight'),
                'ups6_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups6_1_Weight'),
                'ups6_2_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Ups6_2_Weight'),
                'ups6_3_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups6_3_Weight'),
                'downs6_1_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs6_1_Weight'),
                'downs6_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Downs6_2_Weight'),
                'downs6_3_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs6_3_Weight'),
                'ups7_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups7_1_Weight'),
                'ups7_2_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Ups7_2_Weight'),
                'ups7_3_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups7_3_Weight'),
                'downs7_1_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs6_1_Weight'),
                'downs7_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Downs6_2_Weight'),
                'downs7_3_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs6_3_Weight'),
                'ups8_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups7_1_Weight'),
                'ups8_2_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Ups7_2_Weight'),
                'ups8_3_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups7_3_Weight'),
                'downs8_1_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs6_1_Weight'),
                'downs8_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Downs6_2_Weight'),
                'downs8_3_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs6_3_Weight'),
                'ups9_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups7_1_Weight'),
                'ups9_2_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Ups7_2_Weight'),
                'ups9_3_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups7_3_Weight'),
                'downs9_1_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs6_1_Weight'),
                'downs9_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Downs6_2_Weight'),
                'downs9_3_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Downs6_3_Weight'),
                'ups10_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups7_1_Weight'),
                'ups10_2_weight': tf.Variable(tf.random_normal([12, 12, 4, 256], stddev=1e-2), name='Ups7_2_Weight'),
                'ups10_3_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-2), name='Ups7_3_Weight'),
                'concat18_weight' : tf.Variable(tf.random_normal([3, 3, 4*10, 3], stddev=1e-2), name='Concat11_Weight'),
            }
            self.biases = {
                'init_conv_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Init_Conv_Bias'),
                'ups1_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups1_1_Bias'),
                'ups1_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups1_2_Bias'),
                'ups1_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups1_3_Bias'),
                'downs1_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs1_1_Bias'),
                'downs1_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs1_2_Bias'),
                'downs1_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs1_3_Bias'),
                'ups2_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_1_Bias'),
                'ups2_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_2_Bias'),
                'ups2_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_3_Bias'),
                'downs2_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_1_Bias'),
                'downs2_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_2_Bias'),
                'downs2_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_3_Bias'),
                'ups3_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_1_Bias'),
                'ups3_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_2_Bias'),
                'ups3_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_3_Bias'),
                'downs3_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_1_Bias'),
                'downs3_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_2_Bias'),
                'downs3_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_3_Bias'),
                'ups4_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_1_Bias'),
                'ups4_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_2_Bias'),
                'ups4_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_3_Bias'),
                'downs4_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_1_Bias'),
                'downs4_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_2_Bias'),
                'downs4_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_3_Bias'),
                'ups5_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_1_Bias'),
                'ups5_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_2_Bias'),
                'ups5_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_3_Bias'),
                'downs5_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_1_Bias'),
                'downs5_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_2_Bias'),
                'downs5_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_3_Bias'),
                'ups6_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_1_Bias'),
                'ups6_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_2_Bias'),
                'ups6_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_3_Bias'),
                'downs6_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_1_Bias'),
                'downs6_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_2_Bias'),
                'downs6_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_3_Bias'),
                'ups7_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_1_Bias'),
                'ups7_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_2_Bias'),
                'ups7_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_3_Bias'),
                'downs7_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_1_Bias'),
                'downs7_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_2_Bias'),
                'downs7_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_3_Bias'),
                'ups8_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_1_Bias'),
                'ups8_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_2_Bias'),
                'ups8_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_3_Bias'),
                'downs8_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_1_Bias'),
                'downs8_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_2_Bias'),
                'downs8_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_3_Bias'),
                'ups9_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_1_Bias'),
                'ups9_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_2_Bias'),
                'ups9_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_3_Bias'),
                'downs9_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_1_Bias'),
                'downs9_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_2_Bias'),
                'downs9_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Downs2_3_Bias'),
                'ups10_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_1_Bias'),
                'ups10_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_2_Bias'),
                'ups10_3_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Ups2_3_Bias'),
                'concat18_bias' : tf.Variable(tf.random_normal([3], stddev=1e-3), name='Concat1_Bias')
            }

        w_iter = iter(self.weights)
        b_iter = iter(self.biases)

        with tf.name_scope("initial_layer"):
            if self.merge :
                self.initial_ = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 256], name = 'input_merge')
                next(w_iter)
                next(b_iter)
            else:
                net = tf.nn.conv2d(self.input_images, self.weights[next(w_iter)], strides = [1,1,1,1], padding = 'SAME') + self.biases[next(b_iter)]
                self.initial_ = tf.nn.relu(net)


        with tf.name_scope("back_projection"):
            up_1 = self.backprojection_upsample(self.initial_, self.weights[next(w_iter)], self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])
            down_1 = self.backprojection_downsample(up_1, self.weights[next(w_iter)], self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])
            up_2 = self.backprojection_upsample(down_1, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            down_2 = self.backprojection_downsample(up_2, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            up_3 = self.backprojection_upsample(down_2, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            down_3 = self.backprojection_downsample(up_3, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            up_4 = self.backprojection_upsample(down_3, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            down_4 = self.backprojection_downsample(up_4, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            up_5 = self.backprojection_upsample(down_4, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            down_5 = self.backprojection_downsample(up_5, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            up_6 = self.backprojection_upsample(down_5, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            down_6 = self.backprojection_downsample(up_6, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            up_7 = self.backprojection_upsample(down_6, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            down_7 = self.backprojection_downsample(up_7, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            up_8 = self.backprojection_upsample(down_7, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            down_8 = self.backprojection_downsample(up_8, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            up_9 = self.backprojection_upsample(down_8, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            down_9 = self.backprojection_downsample(up_9, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            up_10 = self.backprojection_upsample(down_9, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])

            temp = tf.concat([up_1,up_2, up_3, up_4, up_5, up_6, up_7, up_8, up_9, up_10], 3)
            temp = tf.nn.conv2d(temp, self.weights[next(w_iter)], strides=[1, 1, 1, 1], padding='SAME') + self.biases[next(b_iter)]
            temp = tf.nn.relu(temp)


        '''
        with tf.name_scope("residual_layer"):
            jump = temp
            net = self.residual_block(jump, self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])
            net = self.residual_block(net, self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])
            net = self.residual_block(net, self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])
            net = self.residual_block(net, self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])
            net = self.residual_block(net, self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])
            net = self.residual_block(net, self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])



        with tf.name_scope("upsample_layer"):
            net = tf.nn.conv2d(jump + net, self.weights[next(w_iter)], strides=[1, 1, 1, 1], padding='SAME') + self.biases[next(b_iter)]
            net = tf.nn.relu(net)
            output = self.shuffle_operator(net, 2)
        '''
        self.pred = temp
        self.loss = tf.reduce_mean(tf.abs(self.label_images - self.pred))
        self.psnr = tf.image.psnr(self.label_images, self.pred, max_val = 1)
        self.saver = tf.train.Saver()
        total_parameters = 0
        # iterating over all variables

        for variable in tf.trainable_variables():
            local_parameters = 1
            shape = variable.get_shape()  # getting shape of a variable
            for i in shape:
                local_parameters *= i.value  # mutiplying dimension values
            total_parameters += local_parameters
        print("total_parameters : " + str(total_parameters))


    def train(self):

        self.save_loss = np.zeros((self.epoch, 3))
        global_step = tf.Variable(0, trainable=False)
        start_lr = 0.0001
        decay_steps = 20000
        decay_rate = 0.1
        learning_rate = tf.compat.v1.train.natural_exp_decay(start_lr, global_step, decay_steps, decay_rate)
        self.train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss = self.loss, global_step = global_step)
        self.sess.run(tf.initialize_all_variables())
        counter = 0

        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if not(self.merge):
            base_path = 'processed'
            train_HR_file_list = sorted(glob.glob(os.path.join(base_path, 'true_train', '*.npy')))
            train_LR_file_list = sorted(glob.glob(os.path.join(base_path, 'bicubic_train', '*.npy')))
            valid_HR_file_list = sorted(glob.glob(os.path.join(base_path, 'true_val', '*.npy')))
            valid_LR_file_list = sorted(glob.glob(os.path.join(base_path, 'bicubic_val', '*.npy')))

            train_range = int(n_train / self.batch_size)
            val_range = int(n_val / self.batch_size)
            start_time = time.time()


            for ep in range(self.epoch) :   # config.epoch
                sum_err = 0

                for i in range(train_range) :
                    temp_time = time.time()
                    batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, 3))
                    batch_label = np.zeros((self.batch_size, self.label_size, self.label_size, 3))
                    for j in range(self.batch_size):
                        batch_image[j] = np.load(train_LR_file_list[i * self.batch_size + j])
                        batch_label[j] = np.load(train_HR_file_list[i * self.batch_size + j])

                    _, err = self.sess.run([self.train_op, self.loss], feed_dict = {self.input_images : batch_image, self.label_images : batch_label, self.input_batch_size : self.batch_size})
                    print("Epoch : [%2d], count : [%d], time : [%4.4f], train_loss : [%.8f]" % ((ep + 1), counter, time.time() - temp_time, err))
                    counter += 1

                    sum_err += err
                    if counter % 500 == 0:
                        self.save(counter)

                print("Epoch : [%2d], time : [%4.4f], train_loss : [%.8f]" % ((ep + 1), time.time() - start_time, sum_err/train_range))

                self.save_loss[ep][0] = sum_err / train_range
                sum_err = 0
                sum_psnr = 0
                for i in range(val_range) :

                    batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, 3))
                    batch_label = np.zeros((self.batch_size, self.label_size, self.label_size, 3))
                    for j in range(self.batch_size):
                        batch_image[j] = np.load(valid_LR_file_list[i * self.batch_size + j])
                        batch_label[j] = np.load(valid_HR_file_list[i * self.batch_size + j])

                    err, qwer = self.sess.run([self.loss,self.psnr], feed_dict={self.input_images: batch_image, self.label_images: batch_label, self.input_batch_size : self.batch_size})
                    sum_err += err
                    sum_psnr +=qwer

                print("Epoch [%2d] finished, validation loss : [%.8f], psnr : [%.8f]" % ((ep+1), sum_err/val_range, sum_psnr/val_range))
                self.save_loss[ep][1] = sum_err / val_range
                self.save_loss[ep][2] = sum_psnr / val_range



        else :
            train_HR_file_list = sorted(glob.glob(os.path.join('processedx8', 'true_train', '*.npy')))
            train_LR_file_list = sorted(glob.glob(os.path.join('merge_train', '*.npy')))
            valid_HR_file_list = sorted(glob.glob(os.path.join('processedx8', 'true_val', '*.npy')))
            valid_LR_file_list = sorted(glob.glob(os.path.join('merge_val','*.npy')))

            train_range = int(n_train / self.batch_size)
            val_range = int(n_val / self.batch_size)
            start_time = time.time()

            for ep in range(self.epoch) :   # config.epoch
                sum_err = 0

                for i in range(train_range) :
                    temp_time = time.time()
                    batch_label = np.zeros((self.batch_size, self.label_size, self.label_size, 3))
                    for j in range(self.batch_size):
                        batch_label[j] = np.load(train_HR_file_list[i * self.batch_size + j])
                    batch_image = np.load(train_LR_file_list[i])
                    batch_image = batch_image.reshape((16,self.input_size,self.input_size,256))

                    _, err = self.sess.run([self.train_op, self.loss], feed_dict = {self.initial_ : batch_image, self.label_images : batch_label, self.input_batch_size : self.batch_size})
                    print("Epoch : [%2d], count : [%d], time : [%4.4f], train_loss : [%.8f]" % ((ep + 1), counter, time.time() - temp_time, err))
                    counter += 1

                    sum_err += err
                    if counter % 500 == 0:
                        self.save(counter)

                print("Epoch : [%2d], time : [%4.4f], train_loss : [%.8f]" % ((ep + 1), time.time() - start_time, sum_err/train_range))
                self.save_loss[ep][0] = sum_err / train_range

                sum_err = 0
                sum_psnr = 0
                for i in range(val_range):
                    batch_label = np.zeros((self.batch_size, self.label_size, self.label_size, 3))
                    for j in range(self.batch_size):
                        batch_label[j] = np.load(valid_HR_file_list[i * self.batch_size + j])
                    batch_image = np.load(valid_LR_file_list[i])
                    batch_image = batch_image.reshape((16, self.input_size, self.input_size, 256))
                    err, qwer = self.sess.run([self.loss,self.psnr],feed_dict={self.initial_ : batch_image, self.label_images : batch_label, self.input_batch_size : self.batch_size})
                    sum_err += err
                    sum_psnr += qwer

                print("Epoch [%2d] finished, validation loss : [%.8f], psnr : [%.8f]" % ((ep + 1), sum_err / val_range, sum(sum_psnr/val_range)/self.batch_size))
                self.save_loss[ep][1] = sum_err / val_range
                self.save_loss[ep][2] = sum(sum_psnr/val_range)/self.batch_size



        np.save(os.path.join("save_loss","JJongs6_tensorflow_" + str(self.epoch) + '.npy'), self.save_loss)





    def predict(self,image_file,base_path = 'test_image'):

        image_file = os.path.join(base_path, image_file)

        input_image = cv2.cvtColor(cv2.imread(image_file),cv2.COLOR_BGR2RGB)
        input_image = cv2.normalize(input_image.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)

        h, w, _ = input_image.shape
        h = (int)(h / self.input_size)
        w = (int)(w / self.input_size)

        pred_image = np.zeros((h * self.label_size, w * self.label_size, 3))

        for j in range(h):
            for k in range(w):
                patch_image = input_image[j * self.input_size:(j + 1) * self.input_size, k * self.input_size:(k + 1) * self.input_size]
                patch_image = patch_image.reshape((1, self.input_size, self.input_size, 3))
                pred = self.sess.run(self.pred, feed_dict = {self.input_images : patch_image,self.input_batch_size : 1})
                pred = np.clip(pred.reshape((self.label_size, self.label_size, 3)), 0, 1)
                pred_image[j * self.label_size:(j + 1) * self.label_size, k * self.label_size:(k + 1) * self.label_size] = pred

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('input')
        plt.imshow(input_image[:h * self.input_size, :w * self.input_size])
        plt.subplot(1, 2, 2)
        plt.title('pred')
        plt.imshow(pred_image)
        self.pred_image = pred_image


    def save(self,step) :
        model_name = "JJongs6_tensorflow.model"
        model_dir = "%s_%s" % ("JJongs6_tensorflow_scale_x", self.scale)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir) :
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step = step)


    def load(self) :
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("JJongs6_tensorflow_scale_x", self.scale)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path :
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def _psnr(self):

        _base_path = "processed"
        _valid_HR_file_list = sorted(glob.glob(os.path.join(_base_path, 'true_val', '*.npy')))
        _valid_LR_file_list = sorted(glob.glob(os.path.join(_base_path, 'bicubic_val', '*.npy')))

        psnr = 0.0
        for i in range(n_val):

            batch_image = np.zeros((1, self.input_size, self.input_size, 3))
            batch_label = np.zeros((1, self.label_size, self.label_size, 3))
            batch_image[0] = np.load(_valid_LR_file_list[i])
            batch_label[0] = np.load(_valid_HR_file_list[i])

            temp = self.sess.run(self.psnr, feed_dict={self.input_images: batch_image, self.label_images: batch_label, self.input_batch_size: 1})
            psnr += temp

        psnr /= n_val

        return psnr

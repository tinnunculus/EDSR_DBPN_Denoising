import tensorflow as tf
import numpy as np
import os, glob, time, cv2, math
import matplotlib.pyplot as plt

n_train = 49110
n_val = 6250


class JJONGS_MODEL(object):
    def __init__(self, sess, input_size = 50, label_size = 200, batch_size = 16, pretrain = False, scale = 4, epoch = 50,  checkpoint_dir = "jjongs_tensorflow_checkpoint") :

        self.sess = sess
        self.input_size = input_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.scale = scale
        self.epoch = epoch
        self.checkpoint_dir = checkpoint_dir
        self.build_model()


    def residual_block(self, x, weight1, weight2, bias1, bias2):
        y = tf.nn.conv2d(x, weight1, strides = [1,1,1,1], padding ='SAME') + bias1
        y = tf.nn.relu(y)
        y = tf.nn.conv2d(y, weight2, strides = [1,1,1,1], padding = 'SAME') + bias2
        y = y * 0.1
        return x + y

    def backprojection_upsample(self, x, weight1, weight2, weight3, bias1, bias2, bias3):
        temp = tf.nn.conv2d_transpose(x, weight1, output_shape = (self.input_batch_size, 2 * self.input_size, 2 * self.input_size, 128), strides = [1,2,2,1], padding = 'SAME') + bias1
        y = tf.nn.conv2d(temp, weight2, strides = [1,2,2,1], padding = 'SAME') + bias2
        z = y - x
        k = tf.nn.conv2d_transpose(z, weight3, output_shape = (self.input_batch_size, 2 * self.input_size, 2 * self.input_size, 128), strides = [1,2,2,1], padding = 'SAME') + bias3
        return k + temp

    def backprojection_downsample(self, x, weight1, weight2, weight3, bias1, bias2, bias3):
        temp = tf.nn.conv2d(x, weight1, strides = [1,2,2,1], padding = 'SAME') + bias1
        y = tf.nn.conv2d_transpose(temp, weight2, output_shape = (self.input_batch_size, 2 * self.input_size, 2 * self.input_size, 128), strides = [1,2,2,1], padding = 'SAME') + bias2
        z = y - x
        k = tf.nn.conv2d(z, weight3, strides = [1,2,2,1], padding = 'SAME') + bias3
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
                'init_conv_weight' : tf.Variable(tf.random_normal([3, 3, 3, 256], stddev=1e-3), name='Init_Conv_Weight'),
                'res1_1_weight' : tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res1_1_Weight'),
                'res1_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res1_2_Weight'),
                'res2_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res2_1_Weight'),
                'res2_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res2_2_Weight'),
                'res3_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res3_1_Weight'),
                'res3_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res3_2_Weight'),
                'res4_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res4_1_Weight'),
                'res4_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res4_2_Weight'),
                'finish_res_conv_weight' : tf.Variable(tf.random_normal([3, 3, 256, 128], stddev=1e-3), name='Finish_Res_Conv_Weight'),
                'ups1_1_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups1_1_Weight'),
                'ups1_2_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups1_2_Weight'),
                'ups1_3_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups1_3_Weight'),
                'downs1_1_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs1_1_Weight'),
                'downs1_2_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs1_2_Weight'),
                'downs1_3_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs1_3_Weight'),
                'ups2_1_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups2_1_Weight'),
                'ups2_2_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups2_2_Weight'),
                'ups2_3_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups2_3_Weight'),
                'downs2_1_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs2_1_Weight'),
                'downs2_2_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs2_2_Weight'),
                'downs2_3_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs2_3_Weight'),
                'ups3_1_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups3_1_Weight'),
                'ups3_2_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups3_2_Weight'),
                'ups3_3_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups3_3_Weight'),
                'downs3_1_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs3_1_Weight'),
                'downs3_2_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs3_2_Weight'),
                'downs3_3_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs3_3_Weight'),
                'ups4_1_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups4_1_Weight'),
                'ups4_2_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups4_2_Weight'),
                'ups4_3_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups4_3_Weight'),
                'downs4_1_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs4_1_Weight'),
                'downs4_2_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs4_2_Weight'),
                'downs4_3_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs4_3_Weight'),
                'ups5_1_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups5_1_Weight'),
                'ups5_2_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups5_2_Weight'),
                'ups5_3_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups5_3_Weight'),
                'downs5_1_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs5_1_Weight'),
                'downs5_2_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs5_2_Weight'),
                'downs5_3_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs5_3_Weight'),
                'ups6_1_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups6_1_Weight'),
                'ups6_2_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups6_2_Weight'),
                'ups6_3_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups6_3_Weight'),
                'downs6_1_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs6_1_Weight'),
                'downs6_2_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs6_2_Weight'),
                'downs6_3_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Downs6_3_Weight'),
                'ups7_1_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups7_1_Weight'),
                'ups7_2_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups7_2_Weight'),
                'ups7_3_weight': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=1e-3), name='Ups7_3_Weight'),
                'finish_ups_conv_weight': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=1e-3),name='Finish_Ups_Conv_Weight'),
                'shuffle_weight' : tf.Variable(tf.random_normal([3, 3, 256, 3 * 2 * 2], stddev=1e-3), name='Shuffle_Weight')
            }
            self.biases = {
                'init_conv_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Init_Conv_Bias'),
                'res1_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res1_1_Bias'),
                'res1_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res1_2_Bias'),
                'res2_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res2_1_Bias'),
                'res2_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res2_2_Bias'),
                'res3_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res3_1_Bias'),
                'res3_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res3_2_Bias'),
                'res4_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res4_1_Bias'),
                'res4_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res4_2_Bias'),
                'finish_res_conv_bias': tf.Variable(tf.random_normal([128], stddev=1e-3),name='Finish_Res_Conv_Bias'),
                'ups1_1_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups1_1_Bias'),
                'ups1_2_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups1_2_Bias'),
                'ups1_3_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups1_3_Bias'),
                'downs1_1_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs1_1_Bias'),
                'downs1_2_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs1_2_Bias'),
                'downs1_3_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs1_3_Bias'),
                'ups2_1_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups2_1_Bias'),
                'ups2_2_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups2_2_Bias'),
                'ups2_3_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups2_3_Bias'),
                'downs2_1_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs2_1_Bias'),
                'downs2_2_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs2_2_Bias'),
                'downs2_3_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs2_3_Bias'),
                'ups3_1_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups3_1_Bias'),
                'ups3_2_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups3_2_Bias'),
                'ups3_3_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups3_3_Bias'),
                'downs3_1_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs3_1_Bias'),
                'downs3_2_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs3_2_Bias'),
                'downs3_3_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs3_3_Bias'),
                'ups4_1_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups4_1_Bias'),
                'ups4_2_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups4_2_Bias'),
                'ups4_3_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups4_3_Bias'),
                'downs4_1_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs4_1_Bias'),
                'downs4_2_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs4_2_Bias'),
                'downs4_3_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs4_3_Bias'),
                'ups5_1_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups5_1_Bias'),
                'ups5_2_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups5_2_Bias'),
                'ups5_3_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups5_3_Bias'),
                'downs5_1_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs5_1_Bias'),
                'downs5_2_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs5_2_Bias'),
                'downs5_3_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs5_3_Bias'),
                'ups6_1_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups6_1_Bias'),
                'ups6_2_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups6_2_Bias'),
                'ups6_3_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups6_3_Bias'),
                'downs6_1_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs6_1_Bias'),
                'downs6_2_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs6_2_Bias'),
                'downs6_3_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Downs6_3_Bias'),
                'ups7_1_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups7_1_Bias'),
                'ups7_2_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups7_2_Bias'),
                'ups7_3_bias': tf.Variable(tf.random_normal([128], stddev=1e-3), name='Ups7_3_Bias'),
                'finish_ups_conv_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Finish_Ups_Conv_Bias'),
                'shuffle_bias': tf.Variable(tf.random_normal([3 * 2 * 2], stddev=1e-3),name='Shuffle_Bias')
            }

        w_iter = iter(self.weights)
        b_iter = iter(self.biases)

        with tf.name_scope("initial_layer"):
            initial_layer_conv = tf.nn.conv2d(self.input_images, self.weights[next(w_iter)], strides = [1,1,1,1], padding = 'SAME') + self.biases[next(b_iter)]
            initial_layer_act = tf.nn.relu(initial_layer_conv)

        with tf.name_scope("residual_layer"):
            net = self.residual_block(initial_layer_act, self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])
            net = self.residual_block(net, self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])
            net = self.residual_block(net, self.weights[next(w_iter)], self.weights[next(w_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            net = self.residual_block(net, self.weights[next(w_iter)], self.weights[next(w_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            net = initial_layer_act + net
            net = tf.nn.conv2d(net,self.weights[next(w_iter)], strides = [1,1,1,1], padding = 'SAME') + self.biases[next(b_iter)]
            net = tf.nn.relu(net)

        with tf.name_scope("back_projection"):
            up_1 = self.backprojection_upsample(net, self.weights[next(w_iter)], self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])
            down_1 = self.backprojection_downsample(up_1, self.weights[next(w_iter)], self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])
            up_2 = self.backprojection_upsample(down_1, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            temp = (up_1 + up_2)/2
            down_2 = self.backprojection_downsample(temp, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            temp = (down_1 + down_2)/2
            up_3 = self.backprojection_upsample(temp, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            temp = (up_1 + up_2 + up_3)/3
            down_3 = self.backprojection_downsample(temp, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            temp = (down_1 + down_2 + down_3)/3
            up_4 = self.backprojection_upsample(temp, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            temp = (up_1 + up_2+ up_3 + up_4)/4
            down_4 = self.backprojection_downsample(temp, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            temp = (down_1 + down_2 + down_3 + down_4) / 4
            up_5 = self.backprojection_upsample(temp, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            temp = (up_1 + up_2 + up_3 + up_4 + up_5) / 5
            down_5 = self.backprojection_downsample(temp, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            temp = (down_1 + down_2 + down_3 + down_4 + down_5) / 5
            up_6 = self.backprojection_upsample(temp, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            temp = (up_1 + up_2 + up_3 + up_4 + up_5 + up_6) / 6
            down_6 = self.backprojection_downsample(temp, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            temp = (down_1 + down_2 + down_3 + down_4 + down_5 + down_6) / 6
            up_7 = self.backprojection_upsample(temp, self.weights[next(w_iter)], self.weights[next(w_iter)],self.weights[next(w_iter)], self.biases[next(b_iter)],self.biases[next(b_iter)], self.biases[next(b_iter)])
            temp = up_1 + up_2 + up_3 + up_4 + up_5 + up_6 + up_7
            temp = tf.nn.conv2d(temp, self.weights[next(w_iter)], strides=[1, 1, 1, 1], padding='SAME') + self.biases[next(b_iter)]
            temp = tf.nn.relu(temp)

        with tf.name_scope("upsample_layer"):
            in_shuffle_conv = tf.nn.conv2d(temp,self.weights[next(w_iter)], strides = [1,1,1,1], padding = 'SAME') + self.biases[next(b_iter)]
            in_shuffle_act = tf.nn.relu(in_shuffle_conv)
            output = self.shuffle_operator(in_shuffle_act, 2)

        self.pred = output
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

        self.save_loss = np.zeros((self.epoch, 2))


        base_path = 'processed'
        train_HR_file_list = sorted(glob.glob(os.path.join(base_path, 'y_train', '*.npy')))
        train_LR_file_list = sorted(glob.glob(os.path.join(base_path, 'x_train', '*.npy')))
        valid_HR_file_list = sorted(glob.glob(os.path.join(base_path, 'y_val', '*.npy')))
        valid_LR_file_list = sorted(glob.glob(os.path.join(base_path, 'x_val', '*.npy')))

        global_step = tf.Variable(0, trainable=False)
        start_lr = 0.0001
        decay_steps = 70000
        decay_rate = 0.5
        learning_rate = tf.compat.v1.train.natural_exp_decay(start_lr, global_step, decay_steps, decay_rate)
        self.train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss = self.loss, global_step = global_step)
        tf.initialize_all_variables().run()

        counter = 0

        train_range = int(n_train/self.batch_size)
        val_range = int(n_val/self.batch_size)
        start_time = time.time()


        #if self.load():
        #    print(" [*] Load SUCCESS")
        #else:
        #    print(" [!] Load failed...")


        for ep in range(self.epoch) :   # config.epoch
            sum_err = 0

            for i in range(train_range) :

                batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, 3))
                batch_label = np.zeros((self.batch_size, self.label_size, self.label_size, 3))
                for j in range(self.batch_size):
                    batch_image[j] = np.load(train_LR_file_list[i * self.batch_size + j])
                    batch_label[j] = np.load(train_HR_file_list[i * self.batch_size + j])

                temp_time = time.time()
                _, err = self.sess.run([self.train_op, self.loss], feed_dict = {self.input_images : batch_image, self.label_images : batch_label, self.input_batch_size : self.batch_size})
                print("Epoch : [%2d], count : [%d], time : [%4.4f], train_loss : [%.8f]" % ((ep + 1), counter, time.time() - temp_time, err))
                counter += 1

                sum_err += err
                #if counter % 500 == 0:
                    #self.save(counter)

            print("Epoch : [%2d], time : [%4.4f], train_loss : [%.8f]" % ((ep + 1), time.time() - start_time, sum_err/train_range))

            self.save_loss[ep][0] = sum_err / train_range
            sum_err = 0
            for i in range(val_range) :

                batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, 3))
                batch_label = np.zeros((self.batch_size, self.label_size, self.label_size, 3))
                for j in range(self.batch_size):
                    batch_image[j] = np.load(valid_LR_file_list[i * self.batch_size + j])
                    batch_label[j] = np.load(valid_HR_file_list[i * self.batch_size + j])

                err = self.sess.run(self.loss, feed_dict={self.input_images: batch_image, self.label_images: batch_label, self.input_batch_size : self.batch_size})
                sum_err += err

            print("Epoch [%2d] finished, validation loss : [%.8f]" % ((ep+1), sum_err/val_range))
            self.save_loss[ep][1] = sum_err / val_range

        #np.save(os.path.join("save_loss","JJongs1_tensorflow_" + str(self.epoch) + '.npy'), self.save_loss)





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
        model_name = "JJongs1_tensorflow.model"
        model_dir = "%s_%s" % ("JJongs1_tensorflow_scale_x", self.scale)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir) :
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step = step)


    def load(self) :
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("JJongs1_tensorflow_scale_x", self.scale)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path :
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def _psnr(self):

        base_path = "processed"
        valid_HR_file_list = sorted(glob.glob(os.path.join(base_path, 'y_val', '*.npy')))
        valid_LR_file_list = sorted(glob.glob(os.path.join(base_path, 'x_val', '*.npy')))

        psnr = 0.0
        for i in range(n_val):

            batch_image = np.zeros((1, self.input_size, self.input_size, 3))
            batch_label = np.zeros((1, self.label_size, self.label_size, 3))
            batch_image[0] = np.load(valid_LR_file_list[i])
            batch_label[0] = np.load(valid_HR_file_list[i])

            temp = self.sess.run(self.psnr, feed_dict={self.input_images: batch_image, self.label_images: batch_label, self.input_batch_size: 1})
            psnr += temp

        psnr /= n_val

        return psnr

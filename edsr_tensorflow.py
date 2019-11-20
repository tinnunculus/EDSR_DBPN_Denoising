import tensorflow as tf
import numpy as np
import os, glob, time, cv2
import matplotlib.pyplot as plt

n_train = 49110
n_val = 6250


class EDSR_TENSORFLOW(object):
    def __init__(self, sess, input_size = 50, label_size = 200, batch_size = 25, pretrain = False, scale = 4, epoch = 30,  checkpoint_dir = "edsr_tensorflow_checkpoint") :

        self.sess = sess
        self.input_size = input_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.scale = scale
        self.epoch = epoch
        self.checkpoint_dir = checkpoint_dir

        self.build_model()

    def residual_block(self,x,weight1,weight2,bias1,bias2):
        y = tf.nn.conv2d(x, weight1, strides = [1,1,1,1], padding ='SAME') + bias1
        y = tf.nn.relu(y)
        y = tf.nn.conv2d(y, weight2, strides = [1,1,1,1], padding = 'SAME') + bias2
        y = y * 0.1
        return x + y

    def shuffle_operator(self, X, r):
        temp = tf.depth_to_space(X, r)
        return temp

    def build_model(self):
        self.input_images = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3], name = 'input_images')
        self.label_images = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, 3], name = 'output_images')

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
            'res5_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res5_1_Weight'),
            'res5_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res5_2_Weight'),
            'res6_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res6_1_Weight'),
            'res6_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res6_2_Weight'),
            'res7_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res7_1_Weight'),
            'res7_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res7_2_Weight'),
            'res8_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res8_1_Weight'),
            'res8_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res8_2_Weight'),
            'res9_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res9_1_Weight'),
            'res9_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res9_2_Weight'),
            'res10_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res10_1_Weight'),
            'res10_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res10_2_Weight'),
            'res11_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res11_1_Weight'),
            'res11_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res11_2_Weight'),
            'res12_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res12_1_Weight'),
            'res12_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res12_2_Weight'),
            'res13_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res13_1_Weight'),
            'res13_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res13_2_Weight'),
            'res14_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res14_1_Weight'),
            'res14_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res14_2_Weight'),
            'res15_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res15_1_Weight'),
            'res15_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res15_2_Weight'),
            'res16_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res16_1_Weight'),
            'res16_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res16_2_Weight'),
            'res17_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res17_1_Weight'),
            'res17_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res17_2_Weight'),
            'res18_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res18_1_Weight'),
            'res18_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res18_2_Weight'),
            'res19_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res19_1_Weight'),
            'res19_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res19_2_Weight'),
            'res20_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res20_1_Weight'),
            'res20_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res20_2_Weight'),
            'res21_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res21_1_Weight'),
            'res21_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res21_2_Weight'),
            'res22_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res22_1_Weight'),
            'res22_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res22_2_Weight'),
            'res23_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res23_1_Weight'),
            'res23_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res23_2_Weight'),
            'res24_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res24_1_Weight'),
            'res24_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res24_2_Weight'),
            'res25_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res25_1_Weight'),
            'res25_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res25_2_Weight'),
            'res26_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res26_1_Weight'),
            'res26_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res26_2_Weight'),
            'res27_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res27_1_Weight'),
            'res27_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res27_2_Weight'),
            'res28_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res28_1_Weight'),
            'res28_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res28_2_Weight'),
            'res29_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res29_1_Weight'),
            'res29_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res29_2_Weight'),
            'res30_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res30_1_Weight'),
            'res30_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res30_2_Weight'),
            'res31_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res31_1_Weight'),
            'res31_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res31_2_Weight'),
            'res32_1_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res32_1_Weight'),
            'res32_2_weight': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Res32_2_Weight'),
            'out_conv_weight' : tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=1e-3), name='Out_Conv_Weight'),
            'shuffle_weight' : tf.Variable(tf.random_normal([3, 3, 256, 3 * self.scale * self.scale], stddev=1e-3), name='Shuffle_Weight')
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
            'res5_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res5_1_Bias'),
            'res5_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res5_2_Bias'),
            'res6_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res6_1_Bias'),
            'res6_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res6_2_Bias'),
            'res7_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res7_1_Bias'),
            'res7_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res7_2_Bias'),
            'res8_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res8_1_Bias'),
            'res8_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res8_2_Bias'),
            'res9_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res9_1_Bias'),
            'res9_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res9_2_Bias'),
            'res10_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res10_1_Bias'),
            'res10_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res10_2_Bias'),
            'res11_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res11_1_Bias'),
            'res11_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res11_2_Bias'),
            'res12_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res12_1_Bias'),
            'res12_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res12_2_Bias'),
            'res13_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res13_1_Bias'),
            'res13_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res13_2_Bias'),
            'res14_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res14_1_Bias'),
            'res14_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res14_2_Bias'),
            'res15_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res15_1_Bias'),
            'res15_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res15_2_Bias'),
            'res16_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res16_1_Bias'),
            'res16_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res16_2_Bias'),
            'res17_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res17_1_Bias'),
            'res17_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res17_2_Bias'),
            'res18_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res18_1_Bias'),
            'res18_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res18_2_Bias'),
            'res19_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res19_1_Bias'),
            'res19_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res19_2_Bias'),
            'res20_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res20_1_Bias'),
            'res20_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res20_2_Bias'),
            'res21_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res21_1_Bias'),
            'res21_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res21_2_Bias'),
            'res22_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res22_1_Bias'),
            'res22_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res22_2_Bias'),
            'res23_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res23_1_Bias'),
            'res23_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res23_2_Bias'),
            'res24_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res24_1_Bias'),
            'res24_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res24_2_Bias'),
            'res25_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res25_1_Bias'),
            'res25_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res25_2_Bias'),
            'res26_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res26_1_Bias'),
            'res26_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res26_2_Bias'),
            'res27_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res27_1_Bias'),
            'res27_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res27_2_Bias'),
            'res28_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res28_1_Bias'),
            'res28_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res28_2_Bias'),
            'res29_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res29_1_Bias'),
            'res29_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res29_2_Bias'),
            'res30_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res30_1_Bias'),
            'res30_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res30_2_Bias'),
            'res31_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res31_1_Bias'),
            'res31_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res31_2_Bias'),
            'res32_1_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res32_1_Bias'),
            'res32_2_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Res32_2_Bias'),
            'out_conv_bias': tf.Variable(tf.random_normal([256], stddev=1e-3), name='Out_Conv_Bias'),
            'shuffle_bias': tf.Variable(tf.random_normal([3 * self.scale * self.scale], stddev=1e-3),name='Shuffle_Bias')
        }

        w_iter = iter(self.weights)
        b_iter = iter(self.biases)

        with tf.name_scope("initial_layer"):
            initial_layer_conv = tf.nn.conv2d(self.input_images, self.weights[next(w_iter)], strides = [1,1,1,1], padding = 'SAME') + self.biases[next(b_iter)]
            initial_layer_act = tf.nn.relu(initial_layer_conv)

        with tf.name_scope("residual_layer"):
            net = self.residual_block(initial_layer_act, self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])
            for i in range(31):
                net = self.residual_block(net, self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])

        with tf.name_scope("finish_layer"):
            finish_res_conv = tf.nn.conv2d(net,self.weights[next(w_iter)], strides = [1,1,1,1], padding = 'SAME') + self.biases[next(b_iter)]
            finish_res_act = tf.nn.relu(finish_res_conv)
            in_subpixel = initial_layer_act + finish_res_act

        with tf.name_scope("upsample_layer"):
            in_shuffle_conv = tf.nn.conv2d(in_subpixel,self.weights[next(w_iter)], strides = [1,1,1,1], padding = 'SAME') + self.biases[next(b_iter)]
            in_shuffle_act = tf.nn.relu(in_shuffle_conv)
            output = self.shuffle_operator(in_shuffle_act,self.scale)

        self.pred = output
        self.loss = tf.reduce_mean(tf.abs(self.label_images - self.pred))
        self.saver = tf.train.Saver()


    def train(self):

        self.save_loss = np.zeros((self.epoch, 2))

        base_path = 'processed'
        train_HR_file_list = sorted(glob.glob(os.path.join(base_path, 'y_train', '*.npy')))
        train_LR_file_list = sorted(glob.glob(os.path.join(base_path, 'x_train', '*.npy')))
        valid_HR_file_list = sorted(glob.glob(os.path.join(base_path, 'y_val', '*.npy')))
        valid_LR_file_list = sorted(glob.glob(os.path.join(base_path, 'x_val', '*.npy')))

        global_step = tf.Variable(0, trainable=False)
        start_lr = 0.0001
        decay_steps = 10000
        decay_rate = 0.5
        learning_rate = tf.compat.v1.train.natural_exp_decay(start_lr, global_step, decay_steps, decay_rate)
        self.train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss = self.loss, global_step = global_step)
        tf.initialize_all_variables().run()

        counter = 0

        train_range = int(n_train/self.batch_size)
        val_range = int(n_val/self.batch_size)
        start_time = time.time()


        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")


        for ep in range(self.epoch) :   # config.epoch
            sum_err = 0

            for i in range(train_range) :

                batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, 3))
                batch_label = np.zeros((self.batch_size, self.label_size, self.label_size, 3))
                for j in range(self.batch_size):
                    batch_image[j] = np.load(train_LR_file_list[i * self.batch_size + j])
                    batch_label[j] = np.load(train_HR_file_list[i * self.batch_size + j])

                temp_time = time.time()
                _, err = self.sess.run([self.train_op, self.loss], feed_dict = {self.input_images : batch_image, self.label_images : batch_label})
                print("Epoch : [%2d], count : [%d], time : [%4.4f], train_loss : [%.8f]" % ((ep + 1), counter, time.time() - temp_time, err))
                counter += 1

                sum_err += err
                if counter % 500 == 0:
                    self.save(counter)

            print("Epoch : [%2d], time : [%4.4f], train_loss : [%.8f]" % ((ep + 1), time.time() - start_time, sum_err/train_range))

            self.save_loss[ep][0] = sum_err / train_range
            sum_err = 0
            for i in range(val_range) :

                batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, 3))
                batch_label = np.zeros((self.batch_size, self.label_size, self.label_size, 3))
                for j in range(self.batch_size):
                    batch_image[j] = np.load(valid_LR_file_list[i * self.batch_size + j])
                    batch_label[j] = np.load(valid_HR_file_list[i * self.batch_size + j])

                err = self.sess.run(self.loss, feed_dict={self.input_images: batch_image, self.label_images: batch_label})
                sum_err += err

            print("Epoch [%2d] finished, validation loss : [%.8f]" % ((ep+1), sum_err/val_range))
            self.save_loss[ep][1] = sum_err / val_range

        np.save(os.path.join("save_loss","edsr_tensorflow_" + str(self.epoch) + '.npy'), self.save_loss)





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
                pred = self.sess.run(self.pred, feed_dict = {self.input_images : patch_image})
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
        model_name = "edsr_tensorflow.model"
        model_dir = "%s_%s" % ("edsr_tensorflow_scale_x", self.scale)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir) :
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step = step)


    def load(self) :
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("edsr_tensorflow_scale_x", self.scale)
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path :
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

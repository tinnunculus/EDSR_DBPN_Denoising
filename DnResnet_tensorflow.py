import tensorflow as tf
import numpy as np
import os, glob, time, cv2
import matplotlib.pyplot as plt

n_train = 49110
n_val = 6250


class DNRESNET_TENSORFLOW(object):
    def __init__(self, sess, input_size = 50, batch_size = 16, pretrain = False, epoch = 35,  checkpoint_dir = "dnresnet_tensorflow_checkpoint") :

        self.sess = sess
        self.input_size = input_size
        self.label_size = input_size
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.epoch = epoch
        self.checkpoint_dir = checkpoint_dir

        self.build_model()

    def dn_residual_block(self,x,weight1,weight2,bias1,bias2):
        y = tf.nn.conv2d(x, weight1, strides = [1,1,1,1], padding ='SAME') + bias1
        y = tf.compat.v1.layers.batch_normalization(y)
        y = tf.nn.relu(y)
        y = tf.nn.conv2d(y, weight2, strides = [1,1,1,1], padding = 'SAME') + bias2
        y = tf.compat.v1.layers.batch_normalization(y)
        y = y * 0.1
        return x + y

    def build_model(self):
        self.input_images = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3], name = 'input_images')
        self.label_images = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3], name = 'output_images')

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
            'out_conv_weight' : tf.Variable(tf.random_normal([3, 3, 256, 3], stddev=1e-3), name='Out_Conv_Weight')
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
            'out_conv_bias': tf.Variable(tf.random_normal([3], stddev=1e-3), name='Out_Conv_Bias'),
        }

        w_iter = iter(self.weights)
        b_iter = iter(self.biases)

        with tf.name_scope("initial_layer"):
            initial_layer_conv = tf.nn.conv2d(self.input_images, self.weights[next(w_iter)], strides = [1,1,1,1], padding = 'SAME') + self.biases[next(b_iter)]
            initial_layer_act = tf.nn.relu(initial_layer_conv)

        with tf.name_scope("residual_layer"):
            net = self.dn_residual_block(initial_layer_act, self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])
            for i in range(31):
                self.input_backproj = self.dn_residual_block(net, self.weights[next(w_iter)], self.weights[next(w_iter)], self.biases[next(b_iter)], self.biases[next(b_iter)])

        with tf.name_scope("finish_layer"):
            finish_res_conv = tf.nn.conv2d(self.input_backproj,self.weights[next(w_iter)], strides = [1,1,1,1], padding = 'SAME') + self.biases[next(b_iter)]
            output = self.input_images + finish_res_conv

        self.pred = output
        qwer = np.array([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0],[0.0, -1.0, 0.0]])
        asdf = np.zeros((3,3,3,3))
        for a in range(3):
            for b in range(3):
                asdf[a][b][0][0] = qwer[a][b]
                asdf[a][b][1][1] = qwer[a][b]
                asdf[a][b][2][2] = qwer[a][b]
        sharpen_filter = tf.constant(asdf,dtype=tf.float32)
        self.sharpen_image = tf.nn.conv2d(self.label_images, sharpen_filter, strides = [1,1,1,1], padding = 'SAME')
        self.loss = tf.reduce_mean(tf.abs(self.label_images - self.pred))
        self.saver = tf.train.Saver()


    def train(self):

        self.save_loss = np.zeros((self.epoch, 3))

        base_path = 'processed'
        train_LR_file_list = sorted(glob.glob(os.path.join(base_path, 'bicubic_train', '*.npy')))
        train_LR_wild_w1_file_list = sorted(glob.glob(os.path.join(base_path, 'wild_w1_train', '*.npy')))
        train_LR_wild_w2_file_list = sorted(glob.glob(os.path.join(base_path, 'wild_w2_train', '*.npy')))
        train_LR_wild_w3_file_list = sorted(glob.glob(os.path.join(base_path, 'wild_w3_train', '*.npy')))
        train_LR_wild_w4_file_list = sorted(glob.glob(os.path.join(base_path, 'wild_w4_train', '*.npy')))
        train_LR_difficult_file_list = sorted(glob.glob(os.path.join(base_path, 'difficult_train', '*.npy')))
        valid_LR_file_list = sorted(glob.glob(os.path.join(base_path, 'bicubic_val', '*.npy')))
        valid_LR_wild_file_list = sorted(glob.glob(os.path.join(base_path, 'wild_val', '*.npy')))
        valid_LR_difficult_file_list = sorted(glob.glob(os.path.join(base_path, 'difficult_val', '*.npy')))

        global_step = tf.Variable(0, trainable=False)
        start_lr = 0.0001
        decay_steps = 10000
        decay_rate = 0.5
        learning_rate = tf.compat.v1.train.natural_exp_decay(start_lr, global_step, decay_steps, decay_rate)
        self.train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss = self.loss, global_step = global_step)
        self.sess.run(tf.initialize_all_variables())

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

                temp_time = time.time()
                batch_image = np.zeros((self.batch_size, self.input_size, self.input_size, 3))
                batch_label = np.zeros((self.batch_size, self.input_size, self.input_size, 3))


                # wild1
                for j in range(self.batch_size):
                    batch_image[j] = np.load(train_LR_wild_w1_file_list[i * self.batch_size + j])
                    batch_label[j] = np.load(train_LR_file_list[i * self.batch_size + j])

                self.temp_sharp = self.sess.run(self.sharpen_image, feed_dict = {self.label_images : batch_label})
                self.temp_sharp[:,:,0,:] = batch_label[:,:,0,:]
                self.temp_sharp[:,:,49,:] = batch_label[:,:,49,:]
                self.temp_sharp[:,0,:,:] = batch_label[:,0,:,:]
                self.temp_sharp[:,49,:,:] = batch_label[:,49,:,:]
                _, err = self.sess.run([self.train_op, self.loss], feed_dict = {self.input_images : batch_image, self.label_images : self.temp_sharp})

                print("Epoch : [%2d], count : [%d], time : [%4.4f], train_loss : [%.8f]" % ((ep + 1), counter, time.time() - temp_time, err))

                # wild2
                for j in range(self.batch_size):
                    batch_image[j] = np.load(train_LR_wild_w2_file_list[i * self.batch_size + j])
                    batch_label[j] = np.load(train_LR_file_list[i * self.batch_size + j])

                self.temp_sharp = self.sess.run(self.sharpen_image, feed_dict={self.label_images: batch_label})
                self.temp_sharp[:, :, 0, :] = batch_label[:, :, 0, :]
                self.temp_sharp[:, :, 49, :] = batch_label[:, :, 49, :]
                self.temp_sharp[:, 0, :, :] = batch_label[:, 0, :, :]
                self.temp_sharp[:, 49, :, :] = batch_label[:, 49, :, :]
                _, err = self.sess.run([self.train_op, self.loss],feed_dict={self.input_images: batch_image, self.label_images: self.temp_sharp})

                print("Epoch : [%2d], count : [%d], time : [%4.4f], train_loss : [%.8f]" % (
                (ep + 1), counter, time.time() - temp_time, err))

                # wild3
                for j in range(self.batch_size):
                    batch_image[j] = np.load(train_LR_wild_w3_file_list[i * self.batch_size + j])
                    batch_label[j] = np.load(train_LR_file_list[i * self.batch_size + j])

                self.temp_sharp = self.sess.run(self.sharpen_image, feed_dict={self.label_images: batch_label})
                self.temp_sharp[:, :, 0, :] = batch_label[:, :, 0, :]
                self.temp_sharp[:, :, 49, :] = batch_label[:, :, 49, :]
                self.temp_sharp[:, 0, :, :] = batch_label[:, 0, :, :]
                self.temp_sharp[:, 49, :, :] = batch_label[:, 49, :, :]
                _, err = self.sess.run([self.train_op, self.loss],feed_dict={self.input_images: batch_image, self.label_images: self.temp_sharp})

                print("Epoch : [%2d], count : [%d], time : [%4.4f], train_loss : [%.8f]" % (
                (ep + 1), counter, time.time() - temp_time, err))

                # wild4
                for j in range(self.batch_size):
                    batch_image[j] = np.load(train_LR_wild_w4_file_list[i * self.batch_size + j])
                    batch_label[j] = np.load(train_LR_file_list[i * self.batch_size + j])

                self.temp_sharp = self.sess.run(self.sharpen_image, feed_dict={self.label_images: batch_label})
                self.temp_sharp[:, :, 0, :] = batch_label[:, :, 0, :]
                self.temp_sharp[:, :, 49, :] = batch_label[:, :, 49, :]
                self.temp_sharp[:, 0, :, :] = batch_label[:, 0, :, :]
                self.temp_sharp[:, 49, :, :] = batch_label[:, 49, :, :]
                _, err = self.sess.run([self.train_op, self.loss],feed_dict={self.input_images: batch_image, self.label_images: self.temp_sharp})

                print("Epoch : [%2d], count : [%d], time : [%4.4f], train_loss : [%.8f]" % (
                (ep + 1), counter, time.time() - temp_time, err))


                # difficult
                for j in range(self.batch_size):
                    batch_image[j] = np.load(train_LR_difficult_file_list[i * self.batch_size + j])
                    batch_label[j] = np.load(train_LR_file_list[i * self.batch_size + j])

                self.temp_sharp = self.sess.run(self.sharpen_image, feed_dict={self.label_images: batch_label})
                self.temp_sharp[:, :, 0, :] = batch_label[:, :, 0, :]
                self.temp_sharp[:, :, 49, :] = batch_label[:, :, 49, :]
                self.temp_sharp[:, 0, :, :] = batch_label[:, 0, :, :]
                self.temp_sharp[:, 49, :, :] = batch_label[:, 49, :, :]
                _, err = self.sess.run([self.train_op, self.loss],feed_dict={self.input_images: batch_image, self.label_images: self.temp_sharp})

                print("Epoch : [%2d], count : [%d], time : [%4.4f], train_loss : [%.8f]" % (
                (ep + 1), counter, time.time() - temp_time, err))

                counter += 1
                if counter % 500 == 0:
                    self.save(counter)

    def save(self,step) :
        model_name = "dnresnet_wild_tensorflow.model"
        model_dir = "dnresnet_wild_scale"
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir) :
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step = step)


    def load(self) :
        print(" [*] Reading checkpoints...")
        model_dir = "dnresnet_wild_scale"
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path :
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

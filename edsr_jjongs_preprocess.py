from edsr_tensorflow import EDSR_TENSORFLOW
import numpy as np
import glob, os
import tensorflow as tf

base_path = 'processedx8'
train_LR_file_list = sorted(glob.glob(os.path.join(base_path, 'bicubic_train', '*.npy')))
valid_LR_file_list = sorted(glob.glob(os.path.join(base_path, 'bicubic_val', '*.npy')))

counter = 0
n_train = 27958
n_val = 3598
train_range = int(n_train / 16)
val_range = int(n_val / 16)

sess1 = tf.Session()
jj = EDSR_TENSORFLOW(sess1)
jj.sess.run(tf.initialize_all_variables())
jj.load()



#w1
for i in range(train_range):

    batch_image = np.zeros((16, 32, 32, 3))
    for j in range(16):
        batch_image[j] = np.load(train_LR_file_list[i * 16 + j])

    input_backproj = jj.sess.run([jj.input_backproj],feed_dict={jj.input_images : batch_image})
    if int(i / 10) == 0:
        str_num = '000' + str(i)
    elif int(i / 100) == 0:
        str_num = '00' + str(i)
    elif int(i / 1000) == 0:
        str_num = '0' + str(i)
    else:
        str_num = str(i)

    np.save(os.path.join("merge_train", str_num + '.npy'), input_backproj)

for i in range(val_range):

    batch_image = np.zeros((16, 32, 32, 3))
    for j in range(16):
        batch_image[j] = np.load(valid_LR_file_list[i * 16 + j])

    input_backproj = jj.sess.run([jj.input_backproj],feed_dict={jj.input_images : batch_image})
    if int(i / 10) == 0:
        str_num = '000' + str(i)
    elif int(i / 100) == 0:
        str_num = '00' + str(i)
    elif int(i / 1000) == 0:
        str_num = '0' + str(i)
    else:
        str_num = str(i)

    np.save(os.path.join("merge_val", str_num + '.npy'), input_backproj)

#w2
for i in range(train_range):

    batch_image = np.zeros((16, 50, 50, 3))
    for j in range(16):
        batch_image[j] = np.load(train_LR_wild_w2_file_list[i * 16 + j])

    input_backproj = jj.sess.run([jj.input_backproj],feed_dict={jj.input_images : batch_image})
    if int(i / 10) == 0:
        str_num = '000' + str(i)
    elif int(i / 100) == 0:
        str_num = '00' + str(i)
    elif int(i / 1000) == 0:
        str_num = '0' + str(i)
    else:
        str_num = str(i)

    np.save(os.path.join("merge_wild_w2", str_num + '.npy'), input_backproj)


#w3
for i in range(train_range):

    batch_image = np.zeros((16, 50, 50, 3))
    for j in range(16):
        batch_image[j] = np.load(train_LR_wild_w3_file_list[i * 16 + j])

    input_backproj = jj.sess.run([jj.input_backproj],feed_dict={jj.input_images : batch_image})
    if int(i / 10) == 0:
        str_num = '000' + str(i)
    elif int(i / 100) == 0:
        str_num = '00' + str(i)
    elif int(i / 1000) == 0:
        str_num = '0' + str(i)
    else:
        str_num = str(i)

    np.save(os.path.join("merge_wild_w3", str_num + '.npy'), input_backproj)


#w4
for i in range(train_range):

    batch_image = np.zeros((16, 50, 50, 3))
    for j in range(16):
        batch_image[j] = np.load(train_LR_wild_w4_file_list[i * 16 + j])

    input_backproj = jj.sess.run([jj.input_backproj],feed_dict={jj.input_images : batch_image})
    if int(i / 10) == 0:
        str_num = '000' + str(i)
    elif int(i / 100) == 0:
        str_num = '00' + str(i)
    elif int(i / 1000) == 0:
        str_num = '0' + str(i)
    else:
        str_num = str(i)

    np.save(os.path.join("merge_wild_w4", str_num + '.npy'), input_backproj)


#difficult
for i in range(train_range):

    batch_image = np.zeros((16, 50, 50, 3))
    for j in range(16):
        batch_image[j] = np.load(train_LR_difficult_file_list[i * 16 + j])

    input_backproj = jj.sess.run([jj.input_backproj],feed_dict={jj.input_images : batch_image})
    if int(i / 10) == 0:
        str_num = '000' + str(i)
    elif int(i / 100) == 0:
        str_num = '00' + str(i)
    elif int(i / 1000) == 0:
        str_num = '0' + str(i)
    else:
        str_num = str(i)

    np.save(os.path.join("merge_difficult", str_num + '.npy'), input_backproj)

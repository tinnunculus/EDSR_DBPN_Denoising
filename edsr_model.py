import os, cv2, glob
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.transform import pyramid_reduce
from keras.layers import Conv2D, Input, add ,Lambda, BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from Subpixel import Subpixel
from DataGenerator import DataGenerator
from keras import optimizers

class EDSR_MODEL(object):
    def __init__(self, input_size = 50 , label_size = 200, batch_size = 16, pretrain = False):
        self.input_size = input_size
        self.label_size = label_size
        self.scale_size = (int)(label_size / input_size)
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.build_model()

    def residual_block(self, x):
        y = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same', activation='relu')(x)
        y = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding = 'same')(y)
        y = Lambda(lambda y: y * 0.1)(y)
        return add([x, y])

    def build_model(self):
        if not(self.pretrain) :
            inputs = Input(shape=(self.input_size, self.input_size, 3))
            initial_layer = Conv2D(filters=256, kernel_size=7, strides=1, padding='same')(inputs)
            net = self.residual_block(initial_layer)
            for i in range(31):
                net = self.residual_block(net)
            net = Conv2D(filters = 256, kernel_size = 3, strides = 1, padding= 'same')(net)
            net = add([initial_layer,net])
            if self.scale_size == 4:
                outputs = Subpixel(filters = 3, kernel_size = 3, r = 2, padding ='same')(net)
                outputs = Subpixel(filters = 3, kernel_size = 3, r = 2, padding ='same')(outputs)
            else :
                outputs = Subpixel(filters = 3, kernel_size = 3, r = self.scale_size, padding ='same')(net)

            self.model = Model(inputs = inputs, outputs = outputs)
            opt = optimizers.Adam(lr = 0.0001)
            self.model.compile(opt, loss='mae')
            self.model.summary()

        else :
            base_model_path = 'models'
            self.model = load_model(os.path.join(base_model_path, 'model.h5'), custom_objects={'Subpixel': Subpixel})

    def train(self):
            base_path = 'processed'
            x_train_list = sorted(glob.glob(os.path.join(base_path, 'x_train', '*.npy')))
            x_val_list = sorted(glob.glob(os.path.join(base_path, 'x_val', '*.npy')))

            train_gen = DataGenerator(list_IDs=x_train_list, labels=None, batch_size=self.batch_size, dim=(self.input_size, self.input_size), n_channels=3, n_classes=None, shuffle=True)
            val_gen = DataGenerator(list_IDs=x_val_list, labels=None, batch_size=self.batch_size, dim=(self.input_size, self.input_size), n_channels=3, n_classes=None, shuffle=False)
            history = self.model.fit_generator(train_gen, validation_data = val_gen, epochs=50, verbose = 1, callbacks=[ModelCheckpoint('models\model.h5', monitor='val_loss', verbose = 1, save_best_only=True)])

    def predict(self, image_file, base_path = 'test_image'):
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
                pred = self.model.predict(patch_image.reshape((1, self.input_size, self.input_size, 3)))
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


    def psnr(self, img1, img2):
        PIXEL_MAX = 255.0
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        return 20.0 * math.log10(PIXEL_MAX / math.sqrt(mse))
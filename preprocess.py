import os, cv2, glob
import numpy as np

class PREPROCESS(object):
    def __init__(self, base_path = "dataset", target_base_path = "processed"):
        self.base_path = base_path
        self.target_base_path = target_base_path

        self.train_HR_base_path = os.path.join(self.base_path, 'DIV2K_train_HR')
        self.train_LR_base_path = os.path.join(self.base_path, 'DIV2k_train_LR_bicubic')
        self.valid_HR_base_path = os.path.join(self.base_path, 'DIV2K_valid_HR')
        self.valid_LR_base_path = os.path.join(self.base_path, 'DIV2k_valid_LR_bicubic')

        self.target_train_HR_base_path = os.path.join(self.target_base_path, 'y_train')
        self.target_train_LR_base_path = os.path.join(self.target_base_path, 'x_train')
        self.target_valid_HR_base_path = os.path.join(self.target_base_path, 'y_val')
        self.target_valid_LR_base_path = os.path.join(self.target_base_path, 'x_val')

        self.train_HR_file_list = sorted(glob.glob(os.path.join(self.train_HR_base_path, '*.png')))
        self.train_LR_file_list = sorted(glob.glob(os.path.join(self.train_LR_base_path, '*.png')))
        self.valid_HR_file_list = sorted(glob.glob(os.path.join(self.valid_HR_base_path, '*.png')))
        self.valid_LR_file_list = sorted(glob.glob(os.path.join(self.valid_LR_base_path, '*.png')))

    def low_scale_preprocess(self, patch_size = 200, low_scale = 4):
        num = 1
        for i, e in enumerate(self.train_HR_file_list):
            img = cv2.cvtColor(cv2.imread(e),cv2.COLOR_BGR2RGB)
            img = cv2.normalize(img.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)

            h, w, _ = img.shape
            h = (int)(h/patch_size)
            w = (int)(w/patch_size)

            for j in range(h):
                for k in range(w):
                    patch_image = img[j * patch_size : (j + 1) * patch_size, k * patch_size : (k + 1) * patch_size]

                    if int(num / 10) == 0:
                        str_num = '0000' + str(num)
                    elif int(num / 100) == 0:
                        str_num = '000' + str(num)
                    elif int(num / 1000) == 0:
                        str_num = '00' + str(num)
                    elif int(num / 10000) == 0:
                        str_num = '0' + str(num)
                    else :
                        str_num = str(num)

                    np.save(os.path.join(self.target_train_HR_base_path, str_num +'.npy'), patch_image)
                    num = num + 1

        num = 1
        for i, e in enumerate(self.train_LR_file_list):
            img = cv2.cvtColor(cv2.imread(e),cv2.COLOR_BGR2RGB)
            img = cv2.normalize(img.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)

            h, w, _ = img.shape
            s = (int)(patch_size / low_scale)
            h = (int)(h / s)
            w = (int)(w / s)

            for j in range(h):
                for k in range(w):
                    patch_image = img[j * s:(j + 1) * s, k * s:(k + 1) * s]

                    if int(num / 10) == 0:
                        str_num = '0000' + str(num)
                    elif int(num / 100) == 0:
                        str_num = '000' + str(num)
                    elif int(num / 1000) == 0:
                        str_num = '00' + str(num)
                    elif int(num / 10000) == 0:
                        str_num = '0' + str(num)
                    else:
                        str_num = str(num)

                    np.save(os.path.join(self.target_train_LR_base_path, str_num + '.npy'), patch_image)
                    num = num + 1

        num = 1
        for i, e in enumerate(self.valid_HR_file_list):
            img = cv2.cvtColor(cv2.imread(e),cv2.COLOR_BGR2RGB)
            img = cv2.normalize(img.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)

            h, w, _ = img.shape
            h = (int)(h / patch_size)
            w = (int)(w / patch_size)

            for j in range(h):
                for k in range(w):
                    patch_image = img[j * patch_size:(j + 1) * patch_size, k * patch_size:(k + 1) * patch_size]

                    if int(num / 10) == 0:
                        str_num = '000' + str(num)
                    elif int(num / 100) == 0:
                        str_num = '00' + str(num)
                    elif int(num / 1000) == 0:
                        str_num = '0' + str(num)
                    else:
                        str_num = str(num)
                    np.save(os.path.join(self.target_valid_HR_base_path, str_num + '.npy'), patch_image)
                    num = num + 1

        num = 1
        for i, e in enumerate(self.valid_LR_file_list):
            img = cv2.cvtColor(cv2.imread(e),cv2.COLOR_BGR2RGB)
            img = cv2.normalize(img.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)

            h, w, _ = img.shape
            s = (int)(patch_size / low_scale)
            h = (int)(h / s)
            w = (int)(w / s)

            for j in range(h):
                for k in range(w):
                    patch_image = img[j * s:(j + 1) * s, k * s:(k + 1) * s]

                    if int(num / 10) == 0:
                        str_num = '000' + str(num)
                    elif int(num / 100) == 0:
                        str_num = '00' + str(num)
                    elif int(num / 1000) == 0:
                        str_num = '0' + str(num)
                    else:
                        str_num = str(num)

                    np.save(os.path.join(self.target_valid_LR_base_path, str_num + '.npy'), patch_image)
                    num = num + 1
import os, cv2, glob
import numpy as np

class PREPROCESS(object):
    def __init__(self, base_path = "dataset", target_base_path = "processedx8"):
        self.base_path = base_path
        self.target_base_path = target_base_path

        self.train_HR_base_path = os.path.join(self.base_path, 'DIV2K_train_HR')
        self.train_LR_base_path = os.path.join(self.base_path, 'DIV2k_train_LR_bicubic')
        self.train_LR_WILD_base_path = os.path.join(self.base_path, 'DIV2k_train_LR_wild')
        self.train_LR_DIFFICULT_base_path = os.path.join(self.base_path, 'DIV2k_train_LR_difficult')
        self.valid_HR_base_path = os.path.join(self.base_path, 'DIV2K_valid_HR')
        self.valid_LR_base_path = os.path.join(self.base_path, 'DIV2k_valid_LR_bicubic')
        self.valid_LR_WILD_base_path = os.path.join(self.base_path, 'DIV2k_valid_LR_wild')
        self.valid_LR_DIFFICULT_base_path = os.path.join(self.base_path, 'DIV2k_valid_LR_difficult')

        self.target_train_HR_base_path = os.path.join(self.target_base_path, 'true_train')
        self.target_train_LR_base_path = os.path.join(self.target_base_path, 'bicubic_train')
        self.target_valid_HR_base_path = os.path.join(self.target_base_path, 'true_val')
        self.target_valid_LR_base_path = os.path.join(self.target_base_path, 'bicubic_val')
        self.target_train_LR_WILD_w1_base_path = os.path.join(self.target_base_path, 'wild_w1_train')
        self.target_train_LR_WILD_w2_base_path = os.path.join(self.target_base_path, 'wild_w2_train')
        self.target_train_LR_WILD_w3_base_path = os.path.join(self.target_base_path, 'wild_w3_train')
        self.target_train_LR_WILD_w4_base_path = os.path.join(self.target_base_path, 'wild_w4_train')
        self.target_valid_LR_WILD_base_path = os.path.join(self.target_base_path, 'wild_val')
        self.target_train_LR_DIFFICULT_base_path = os.path.join(self.target_base_path, 'difficult_train')
        self.target_valid_LR_DIFFICULT_base_path = os.path.join(self.target_base_path, 'difficult_val')

        self.train_HR_file_list = sorted(glob.glob(os.path.join(self.train_HR_base_path, '*.png')))
        self.train_LR_file_list = sorted(glob.glob(os.path.join(self.train_LR_base_path, '*.png')))
        self.train_LR_WILD_w1_file_list = sorted(glob.glob(os.path.join(self.train_LR_WILD_base_path, '*w1.png')))
        self.train_LR_WILD_w2_file_list = sorted(glob.glob(os.path.join(self.train_LR_WILD_base_path, '*w2.png')))
        self.train_LR_WILD_w3_file_list = sorted(glob.glob(os.path.join(self.train_LR_WILD_base_path, '*w3.png')))
        self.train_LR_WILD_w4_file_list = sorted(glob.glob(os.path.join(self.train_LR_WILD_base_path, '*w4.png')))
        self.train_LR_DIFFICULT_file_list = sorted(glob.glob(os.path.join(self.train_LR_DIFFICULT_base_path, '*.png')))
        self.valid_HR_file_list = sorted(glob.glob(os.path.join(self.valid_HR_base_path, '*.png')))
        self.valid_LR_file_list = sorted(glob.glob(os.path.join(self.valid_LR_base_path, '*.png')))
        self.valid_LR_WILD_file_list = sorted(glob.glob(os.path.join(self.valid_LR_WILD_base_path, '*.png')))
        self.valid_LR_DIFFICULT_file_list = sorted(glob.glob(os.path.join(self.valid_LR_DIFFICULT_base_path, '*.png')))

    def low_scale_preprocess(self, patch_size = 256, low_scale = 8):

        #HR_train
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
                    patch_low = cv2.resize(patch_image, dsize = ((int)(patch_size / low_scale) ,(int)(patch_size / low_scale)))
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
                    np.save(os.path.join(self.target_train_LR_base_path, str_num +'.npy'), patch_low)
                    num = num + 1


        '''
        #LR bicubic_train
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
    
        #LR wild_w1_train
        num = 1
        for i, e in enumerate(self.train_LR_WILD_w1_file_list):
            img = cv2.cvtColor(cv2.imread(e), cv2.COLOR_BGR2RGB)
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

                    np.save(os.path.join(self.target_train_LR_WILD_w1_base_path, str_num + '.npy'), patch_image)
                    num = num + 1

        #LR wild_w2_train
        num = 1
        for i, e in enumerate(self.train_LR_WILD_w2_file_list):
            img = cv2.cvtColor(cv2.imread(e), cv2.COLOR_BGR2RGB)
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

                    np.save(os.path.join(self.target_train_LR_WILD_w2_base_path, str_num + '.npy'), patch_image)
                    num = num + 1

        #LR wild_w3_train
        num = 1
        for i, e in enumerate(self.train_LR_WILD_w3_file_list):
            img = cv2.cvtColor(cv2.imread(e), cv2.COLOR_BGR2RGB)
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

                    np.save(os.path.join(self.target_train_LR_WILD_w3_base_path, str_num + '.npy'), patch_image)
                    num = num + 1

        #LR wild_w4_train
        num = 1
        for i, e in enumerate(self.train_LR_WILD_w4_file_list):
            img = cv2.cvtColor(cv2.imread(e), cv2.COLOR_BGR2RGB)
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

                    np.save(os.path.join(self.target_train_LR_WILD_w4_base_path, str_num + '.npy'), patch_image)
                    num = num + 1



        
        #LR difficult_train
        num = 1
        for i, e in enumerate(self.train_LR_DIFFICULT_file_list):
            img = cv2.cvtColor(cv2.imread(e), cv2.COLOR_BGR2RGB)
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

                    np.save(os.path.join(self.target_train_LR_DIFFICULT_base_path, str_num + '.npy'), patch_image)
                    num = num + 1
        '''
        #HR val
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
                    patch_low = cv2.resize(patch_image, dsize=((int)(patch_size / low_scale), (int)(patch_size / low_scale)))

                    if int(num / 10) == 0:
                        str_num = '000' + str(num)
                    elif int(num / 100) == 0:
                        str_num = '00' + str(num)
                    elif int(num / 1000) == 0:
                        str_num = '0' + str(num)
                    else:
                        str_num = str(num)
                    np.save(os.path.join(self.target_valid_HR_base_path, str_num + '.npy'), patch_image)
                    np.save(os.path.join(self.target_valid_LR_base_path, str_num +'.npy'), patch_low)
                    num = num + 1
        '''
        #LR bicubic_val
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
        
        #LR wild_val
        num = 1
        for i, e in enumerate(self.valid_LR_WILD_file_list):
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

                    np.save(os.path.join(self.target_valid_LR_WILD_base_path, str_num + '.npy'), patch_image)
                    num = num + 1
                    
        #LR difficult_val
        num = 1
        for i, e in enumerate(self.valid_LR_DIFFICULT_file_list):
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

                    np.save(os.path.join(self.target_valid_LR_DIFFICULT_base_path, str_num + '.npy'), patch_image)
                    num = num + 1
        '''
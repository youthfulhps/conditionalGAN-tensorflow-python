
import os
import random
import numpy as np
from datetime import datetime

import sa_utils as utils

class Dataset(object):
    def __init__(self, flags):
        #self.dataset = dataset
        self.flags = flags

        self.image_size =(512, 512)
        self.ori_shape=(512, 512)

        self.train_dir = '/home/mlm09/SAGAN/data/data/train'
        self.test_dir =  self.flags.test_dir

        self.num_train = 0
        self.num_val = 0
        self.num_test = 0

        self._read_data()
        print("num of training images:{}".format(self.num_train))
        print("num of validation images:{}".format(self.num_val))
        print("num of test images:{}".format(self.num_test))

    def _read_data(self):
        if self.flags.is_test:
            self.test_imgs, self.test_labels = utils.get_test_imgs(target_dir=self.test_dir)
            self.test_img_files = utils.all_files_under(os.path.join(self.test_dir,'input'))

            self.num_test = self.test_imgs.shape[0]

        elif not self.flags.is_test:
            random.seed(datetime.now())
            self.train_img_files_tmp, self.train_label_files_tmp = utils.get_img_path(self.train_dir)

            self.train_img_files, self.train_label_files = [],[]
            self.val_img_files, self.val_label_files = [], []
            self.num_train = int(len(self.train_img_files_tmp))
            self.num_val = 123
            self.idx_train = [i for i in range(self.num_train)]
            self.val_index = np.random.choice(self.idx_train, self.num_val, replace=False)
            print(self.val_index)

            for idx in self.val_index:
                self.val_img_files.append(self.train_img_files_tmp[idx])
                self.val_label_files.append(self.train_label_files_tmp[idx])

            self.train_index = [idx for idx in self.idx_train if idx not in self.val_index.tolist()]

            for idx in self.train_index:
                self.train_img_files.append(self.train_img_files_tmp[idx])
                self.train_label_files.append(self.train_label_files_tmp[idx])

            self.val_imgs, self.val_labels = utils.get_val_imgs(self.val_img_files, self.val_label_files)

            self.num_val = len(self.val_imgs)


    def train_next_batch(self, batch_size):
        train_indices = np.random.choice(len(self.train_index), batch_size, replace=True)
        #print(len(self.train_index))
        #print(train_indices)
        train_imgs, train_labels = utils.get_train_batch(self.train_img_files, self.train_label_files, train_indices.astype(np.int32))

        train_imgs = np.expand_dims(train_imgs, axis=3)
        train_labels = np.expand_dims(train_labels, axis=3)

        return train_imgs, train_labels

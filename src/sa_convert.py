
import os
import numpy as np
import time
#matplotlib.use('agg') #in server
import matplotlib.pyplot as plt

from sa_model import CGAN
import sa_utils as utils

import tensorflow as tf
from PIL import Image

class Conversion(object):
    def __init__(self, flags):

        self.sess = tf.Session()
        self.flags = flags
        self.model = CGAN(self.sess, self.flags, (512,512))
        self.saver = tf.train.Saver()

        if self.load_model():
            print('[*] Load_Success!')
        else:
            print('[!] Load_Fail!')


    def convert(self):

        if not self.flags.is_single:

            self.generated = []
            self.img_files = utils.all_files_under(os.path.join(self.flags.test_dir,'input'))
            self.num_data = len(self.img_files)
            self.x_imgs = utils.imagefiles2arrs(self.img_files)

            for i in range(self.num_data):
                samples = self.model.sample_imgs(np.expand_dims(np.expand_dims(self.x_imgs[i,...]/255,axis=0),axis=3))
                samples = np.squeeze(samples, axis=(0,3))
                self.generated.append(samples)

            for i in range(len(self.generated)):
                Image.fromarray((self.generated[i]*255).astype(np.uint8)).save(os.path.join(self.flags.output_dir,'test_res_{}.png'.format(i)))

        elif self.flags.is_single:

            self.generated = []
            self.img_file = self.flags.input_file
            self.x_img = utils.imagefiles2arrs(self.img_file)
            self.x_img = utils.pad_imgs(self.x_img, (640,640))

            self.sample = self.model.sample_imgs(np.expand_dims(self.x_img),axis=0)
            self.sample = np.squeeze(sample, axis=(0,3))

            Image.fromarray((self.sample*255).astype(np.uint8)).save(os.path.join(self.flags.output_dir, 'test_res_{}.png'.format(i)))


    def load_model(self):
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(self.flags.model_dir)
        all_models = ckpt.all_model_checkpoint_paths
        if ckpt and all_models:
            ckpt_name = os.path.basename(all_models[self.flags.mn])
            self.saver.restore(self.sess, os.path.join(self.flags.model_dir,ckpt_name))

            print('===================================')
            print('           Model saved             ')
            print('===================================')

            return True
        else:
            return False

if __name__=='__main__':
    Conversion(tf.flags.FLAGS)

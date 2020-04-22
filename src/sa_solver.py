
import os
import time
import collections
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from PIL import Image

from sa_dataset import Dataset
import sa_Tensorflow_utils as tf_utils
import sa_utils as utils
from sa_model import CGAN


class Solver(object):
    def __init__(self, flags):
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        self.sess = tf.Session()

        self.flags = flags
        self.dataset = Dataset(self.flags)
        self.model = CGAN(self.sess, self.flags, self.dataset.image_size)

        self.score = 0.    #best auc x
        self._make_folders()

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        #tf_utils.show_all_variables()

    def _make_folders(self):
        self.model_out_dir = "{}/model_gan*{}+seg*{}".format(self.flags.output_dir, self.flags.lambda2 ,self.flags.lambda1)

        if not os.path.isdir(self.model_out_dir):
            os.makedirs(self.model_out_dir)

        if self.flags.is_test:
            self.img_out_dir = "{}/seg_result_gan*{}+seg*{}/{}".format(self.flags.output_dir, self.flags.lambda2 ,self.flags.lambda1, self.flags.fn)

            self.auc_out_dir = "{}/score_gan*{}+seg*{}".format(self.flags.output_dir, self.flags.lambda2 ,self.flags.lambda1)

            if not os.path.isdir(self.img_out_dir):
                os.makedirs(self.img_out_dir)
            if not os.path.isdir(self.auc_out_dir):
                os.makedirs(self.auc_out_dir)

        elif not self.flags.is_test:
            self.sample_out_dir = "{}/sample_gan*{}+seg*{}".format(self.flags.output_dir, self.flags.lambda2 ,self.flags.lambda1)

            if not os.path.isdir(self.sample_out_dir):
                os.makedirs(self.sample_out_dir)



    def train(self):
        for iter_time in range(0, self.flags.iters+1, self.flags.train_interval):
            self.sample(iter_time)

            for iter_ in range(1, self.flags.train_interval+1):
                x_imgs, y_imgs = self.dataset.train_next_batch(batch_size=self.flags.batch_size)
                d_loss = self.model.train_dis(x_imgs, y_imgs, iter_time)
                self.print_info(iter_time +iter_, 'd_loss', d_loss)

            for iter_ in range(1, self.flags.train_interval+1):
                x_imgs, y_imgs = self.dataset.train_next_batch(batch_size=self.flags.batch_size)
                g_loss = self.model.train_gen(x_imgs, y_imgs, iter_time)
                self.print_info(iter_time+iter_,'g_loss', g_loss)

            #auc_sum = self.eval(iter_time, phase='train')
            #if self.best_auc_sum < auc_sum:
            #    self.best_auc_sum = auc_sum
            #    self.save_model(iter_time)
            self.score = self.eval(iter_time, phase='train')
            if np.mod(iter_time, self.flags.save_freq)==0:
                self.save_model(iter_time)

    def test(self):
        if self.load_model():
            print('[*] Load Success!\n')
            self.eval(phase='test')
        else:
            print('[!] Load Failed!\n')



    def sample(self, iter_time):
        if np.mod(iter_time, self.flags.sample_freq) == 0:
            #idx = np.random.choice(self.dataset.num_val,1, replace=False)
            x_imgs, samples=[], []
            for i in range(self.dataset.num_val):
                x_imgs.append(self.dataset.val_imgs[i])

            for j in range(len(x_imgs)):
                sample = self.model.sample_imgs(np.expand_dims(x_imgs[j],axis=0))
                samples.append(sample)
            #x_imgs, y_imgs = self.dataset.val_imgs, self.dataset.val_labels
            #samples = self.model.sample_imgs(x_imgs)

            self.plot(samples, save_file="{}/".format(self.sample_out_dir), phase='train')


    def plot(self, samples, idx=None, save_file=None, phase='train'):
        '''
        if phase=='train':
            for idx_ in range(len(samples)):
                Image.fromarray(np.squeeze(samples[idx_]*255, axis=(0,3))
                .astype(np.uint8)).save(os.path.join(save_file,'{}_{}.png'.format(os.path.basename(str(self.dataset.val_index[idx_])))))
            #index1 = self.dataset.train_index[idx[0]]
            #Image.fromarray(np.squeeze(samples*255).astype(np.uint8)).save(os.path.join(
                #save_file, '{}_{}.png'.format(os.path.basename(self.dataset.train_img_files[index1][:-4]),iter_time)))
        '''
        if phase=='test':
                Image.fromarray(np.asarray(samples[idx]*255).astype(np.uint8)).save(os.path.join(save_file,'{}.png'.format(str(idx))))


    def print_info(self, iter_time, name, loss):
        if np.mod(iter_time, self.flags.print_freq) == 0:
            ord_output = collections.OrderedDict([(name, loss),
                                                  ('discriminator', self.flags.discriminator),
                                                  ('train_interval', np.float32(self.flags.train_interval)),
                                                  ('gpu_index', self.flags.gpu_index)])
            utils.print_metrics(iter_time, ord_output)


    def eval(self, iter_time=0, phase='train'):
        total_time, score = 0., 0.
        if np.mod(iter_time, self.flags.eval_freq)==0:
            num_data, imgs, labels = None, None, None
            if phase=='train':
                num_data = self.dataset.num_val
                imgs = self.dataset.val_imgs
                labels = self.dataset.val_labels

            elif phase=='test':
                num_data = self.dataset.num_test
                imgs = self.dataset.test_imgs
                labels = self.dataset.test_labels

            generated = []
            #num_eval = np.random.choice(num_data, 1, replace=False)
            start_time = time.time()
            for iter_ in range(num_data):
                x_img = imgs[iter_]
                y_img = labels[iter_]
                x_img = np.expand_dims(x_img, axis=0)
                generated_label = self.model.sample_imgs(x_img)
                generated_label = np.squeeze(generated_label, axis=(0,3))
                y_img = np.squeeze(y_img, axis=2)
                for i in range(len(generated_label[0])):
                    for j in range(len(generated_label[1])):
                        if y_img[i][j]==0:
                            generated_label[i][j]=0

                generated.append(generated_label)
            total_time += (time.time() - start_time)
            #start_time = time.time()

            #generated_label = self.model.sample_imgs(x_img)
            #total_time += (time.time() - start_time)
            #generated.append(np.squeeze(generated_label, axis=(0,3)))

            #generated = np.asarray(generated)
            #print(generated.shape)



            score = self.measure(generated, labels, num_data, iter_time, phase, total_time)

            if phase=='test':
                print(num_data)
                for idx in range(num_data):

                    self.plot(generated, idx, save_file=self.img_out_dir, phase='test')


        return score


    def measure(self, generated, label, num_data, iter_time, phase, total_time):
        avg_pt = (total_time/num_data)*1000
        n=len(generated)
        print(n)
        sum_psnr =0.
        sum_ssim =0.
        score=0.
        for i in range(n):
            psnr = utils.compare_psnr_(generated[i], label[i,...,0])
            ssim = utils.compare_ssim_(generated[i], label[i,...,0])
            #print('{}th psnr, ssim:{}, {}'.format(i, psnr, ssim))
            sum_psnr+=psnr
            sum_ssim+=ssim


        psnr = sum_psnr/n
        ssim = sum_ssim/n
        #generated = np.array([generated[i,...].flatten() for i in range(n)])
        #label = np.array([label[i,...,0].flatten() for i in range(n)])
        #psnr = utils.compare_psnr_(generated, label)
        #ssim = utils.compare_ssim_(generated, label)

        score = psnr + ssim

        ord_output = collections.OrderedDict([('psnr',psnr), ('ssim',ssim), ('score',score),('avg_pt',avg_pt)])

        utils.print_metrics(iter_time, ord_output)

        if phase =='train':
            self.model.measure_assign(psnr, ssim, score, iter_time)

        return score


    def save_model(self, iter_time):
        self.model.best_score_assign(self.score)

        #model_name = "iter_{}_psnr+ssim_{:.3}".format(iter_time, self.score)
        model_name = "iter_{}_psnr+ssim={:.3}".format(iter_time, self.score)
        self.saver.save(self.sess, os.path.join(self.model_out_dir, model_name))
        print('===================================================')
        print('                     Model saved!                  ')
        print(' Score: {:.3}'.format(self.score))
        print('===================================================\n')


    def load_model(self):
        print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.flags.model_dir)
        all_models = ckpt.all_model_checkpoint_paths
        if ckpt and all_models:
            ckpt_name = os.path.basename(all_models[self.flags.mn])
            self.saver.restore(self.sess, os.path.join(self.flags.model_dir, ckpt_name))

            #self.score = self.sess.run(self.model.score)
            print('====================================================')
            print('                     Model saved!                   ')
            print(' Score: {:.3}'.format(self.score))
            print('====================================================')

            return True
        else:
            return False



'''
    def load_model(self):
        print('[*] Reading checkpoint...')

        #ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        #if ckpt and ckpt.all_model_checkpoint_paths1:
        #ckpt_name = os.path.basename(ckpt.model_checkpoint_paths1)
        self.saver.restore(self.sess, os.path.join(self.model_out_dir,"iter_22400_psnr+ssim_0.0"))

        print('===================================================')
        print('                     Model saved!                  ')
        #print(' Score: {:.3}'.format(self.score))
        print('===================================================\n')

        return True

        else:
            return False


    def load_model(self):
        print(' [*] Reading checkpoint...')

        ckpt = tf.train.get_checkpoint_state(self.model_out_dir)
        if ckpt and ckpt.all_model_checkpoint_paths:
            ckpt_name = os.path.basename(ckpt.all_model_checkpoint_paths)
            self.saver.restore(self.sess, os.path.join(self.model_out_dir, ckpt_name))

            #self.score = self.sess.run(self.model.score)
        print('====================================================')
        print('                     Model saved!                   ')
        print(' Score: {:.3}'.format(self.score))
        print('====================================================')

        return True
        else:
            return False
'''


import os
import sys

import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate
from skimage.measure import compare_ssim, compare_psnr

#skleran psnr, ssim


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname)
                         for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames


def get_files(data_path):
    img_dir = os.path.join(data_path, "input")
    label_dir = os.path.join(data_path, "label")

    img_files = all_files_under(img_dir, extension=".png")
    label_files = all_files_under(label_dir, extension=".png")

    return img_files, label_files


def get_img_path(target_dir):
    #img_files, label_files = None, None

    img_files, label_files = get_files(target_dir)

    return img_files, label_files

def image_shape(filename):
    img = Image.open(filename)
    img_arr = np.asarray(img)
    img_shape = img_arr.shape
    return img_shape


def imagefiles2arrs(filenames):
    img_shape = image_shape(filenames[0])
    images_arr = None

    if len(img_shape)==2:
        images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1]), dtype=np.float32)

    for file_index in range(len(filenames)):
        img = Image.open(filenames[file_index])
        images_arr[file_index] = np.asarray(img).astype(np.float32)

    return images_arr


def get_test_imgs(target_dir):

    img_files, label_files = get_files(target_dir)

    input_imgs = imagefiles2arrs(img_files) /255
    label_imgs = imagefiles2arrs(label_files) /255

    input_imgs = np.expand_dims(input_imgs, -1)
    label_imgs = np.expand_dims(label_imgs, -1)

    return input_imgs, label_imgs


def get_val_imgs(img_files, label_files):

    input_imgs = imagefiles2arrs(img_files)/255
    label_imgs = imagefiles2arrs(label_files)/255

    all_input_imgs = [input_imgs]
    all_label_imgs = [label_imgs]

    flipped_input_imgs_lr = input_imgs[:,:,::-1]
    flipped_label_imgs_lr = label_imgs[:,:,::-1]

    all_input_imgs.append(flipped_input_imgs_lr)
    all_label_imgs.append(flipped_label_imgs_lr)

    flipped_input_imgs_ul = input_imgs[:,::-1,:]
    flipped_label_imgs_ul = label_imgs[:,::-1,:]

    all_input_imgs.append(flipped_input_imgs_ul)
    all_label_imgs.append(flipped_label_imgs_ul)

    input_imgs = np.concatenate(all_input_imgs, axis=0)
    label_imgs = np.concatenate(all_label_imgs, axis=0)

    input_imgs = np.expand_dims(input_imgs, -1)
    label_imgs = np.expand_dims(label_imgs, -1)
    print("Get Validation Images SUCCESS!")

    return input_imgs, label_imgs

def get_train_batch(train_img_files, train_label_files, train_indices):
    batch_size = len(train_indices)

    batch_input_files, batch_label_files =[], []
    for _, idx in enumerate(train_indices):
        batch_input_files.append(train_img_files[idx])
        batch_label_files.append(train_label_files[idx])

    input_imgs = imagefiles2arrs(batch_input_files)/255
    label_imgs = imagefiles2arrs(batch_label_files)/255

    for idx in range(batch_size):
        if np.random.random()<0.33:
            input_imgs[idx] = input_imgs[idx,:,::-1]
            label_imgs[idx] = label_imgs[idx,:,::-1]
            #input_imgs = np.expand_dims(input_imgs, -1)
            #label_imgs = np.expand_dims(label_imgs, -1)
        elif np.random.random()>=0.33 and np.random.random()<0.66:
            input_imgs[idx] = input_imgs[idx,::-1,:]
            label_imgs[idx] = label_imgs[idx,::-1,:]

    #input_imgs = np.expand_dims(input_imgs, -1)
    #label_imgs = np.expand_dims(label_imgs, -1)

    return input_imgs, label_imgs


def compare_ssim_(generated_img, label_img):
    ssim = compare_ssim(generated_img, label_img)
    return ssim

def compare_psnr_(generated_img, label_img):
    psnr = compare_psnr(generated_img, label_img)
    return psnr

def print_metrics(itr, kargs):
    print("*** Iteration {}===>".format(itr))
    for name, value in kargs.items():
        print("{} : {:.6}, ".format(name, value))
    print("")
    sys.stdout.flush()

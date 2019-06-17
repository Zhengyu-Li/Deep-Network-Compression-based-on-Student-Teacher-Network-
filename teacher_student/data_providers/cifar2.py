import tempfile
import os
import h5py
import pickle
import random
import numpy as np
from .data_providers2 import ImagesDataSet, DataProvider

'''
Please note that, all the variables named with "block0", "block1" or "logits" are intermediate outputs collected from teacher and related with our training strategy.
You can change them according to your own design.
'''

def augment_image(image, pad):
    """Perform zero padding, randomly crop image to original size,
    maybe mirror horizontally"""
    flip = random.getrandbits(1)
    if flip:
        image = image[:, ::-1, :]
    init_shape = image.shape
    new_shape = [init_shape[0] + pad * 2,
                 init_shape[1] + pad * 2,
                 init_shape[2]]
    zeros_padded = np.zeros(new_shape)
    zeros_padded[pad:init_shape[0] + pad, pad:init_shape[1] + pad, :] = image
    # randomly crop to original size
    init_x = np.random.randint(0, pad * 2)
    init_y = np.random.randint(0, pad * 2)
    cropped = zeros_padded[
        init_x: init_x + init_shape[0],
        init_y: init_y + init_shape[1],
        :]
    return cropped


def augment_all_images(initial_images, pad):
    new_images = np.zeros(initial_images.shape)
    for i in range(initial_images.shape[0]):
        new_images[i] = augment_image(initial_images[i], pad=4)
    return new_images

def read_cifar(filenames):
    f = h5py.File(filenames, 'r')
    images = f['images'][:]
    labels = f['labels'][:]
    block0 = f['block0'][:]
    block1 = f['block1'][:]
    block2 = f['block2'][:]
    logits = f['logits'][:]
    f.close()
    return images, labels, block0, block1, logits

class CifarDataSet(ImagesDataSet):
    def __init__(self, images, labels, block0, block1, logits, n_classes, shuffle, normalization,
                 augmentation):
        """
        Args:
            images: 4D numpy array
            labels: 2D or 1D numpy array
            n_classes: `int`, number of cifar classes - 10 or 100
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            augmentation: `bool`
        """
        if shuffle is None:
            self.shuffle_every_epoch = False
        elif shuffle == 'once_prior_train':
            self.shuffle_every_epoch = False
            images, labels = self.shuffle_images_and_labels(images, labels)
        elif shuffle == 'every_epoch':
            self.shuffle_every_epoch = False
        else:
            raise Exception("Unknown type of shuffling")

        self.images = images
        self.labels = labels
        self.block0 = block0
        self.block1 = block1
        self.block2 = block2
        self.logits = logits #add
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.normalization = normalization
        self.images = images
        self._train_batch_counter = 0
        self._test_batch_counter = 0
        self.start_new_epoch()

    def start_new_epoch(self, is_training=True):
        if (is_training==True):
            self._train_batch_counter = 0
            if self.shuffle_every_epoch:
                images, labels, block0_out, block1_out,  block2_out, logits_out = self.shuffle_images_and_labels( \
                    self.images, self.labels, self.block0, self.block1, self.block2, self.logits) # block2_out
            else:
                images, labels, block0_out, block1_out,  block2_out, logits_out = \
                    self.images, self.labels,  self.block0, self.block1, self.block2, self.logits #block2_out
        else:
            self._test_batch_counter = 0
            images, labels, block0_out, block1_out,  block2_out, logits_out = \
                self.images, self.labels,  self.block0, self.block1, self.block2, self.logits #block2_out
        if self.augmentation:
            images = augment_all_images(images, pad=4)
        self.epoch_images = images
        self.epoch_labels = labels
        self.epoch_block0 = block0_out
        self.epoch_block1 = block1_out
        self.epoch_block2 = block2_out
        self.epoch_logits = logits_out #add

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size, is_training):
        if is_training==True:
            start = self._train_batch_counter * batch_size
            end = (self._train_batch_counter + 1) * batch_size
            self._train_batch_counter += 1
            print("train_counter:%d\n",self._train_batch_counter)	
            if (self._train_batch_counter>=50000//batch_size):
                end = -1
                self.start_new_epoch(is_training=True)

            images_slice = self.epoch_images[start: end]
            labels_slice = self.epoch_labels[start: end]
            block0_slice = self.epoch_block0[start: end]
            block1_slice = self.epoch_block1[start: end]
            block2_slice = self.epoch_block2[start: end]
            logits_slice = self.epoch_logits[start: end]

        else:
            start = self._test_batch_counter * batch_size
            end = (self._test_batch_counter + 1) * batch_size
            self._test_batch_counter += 1
            print("test_counter:%d\n",self._test_batch_counter) 
            if (self._test_batch_counter>=10000//batch_size):
                end = -1
                self.start_new_epoch(is_training=False)

            images_slice = self.epoch_images[start: end]
            labels_slice = self.epoch_labels[start: end]
            block0_slice = 0
            block1_slice = 0
            block2_slice = 0
            logits_slice = 0

        return images_slice, labels_slice, block0_slice, block1_slice, block2_slice, logits_slice

class CifarDataProvider(DataProvider):
    """Abstract class for cifar readers"""

    def __init__(self, save_path=None, validation_set=None,
                 validation_split=None, shuffle=None, normalization=None,
                 one_hot=True, **kwargs):
        """
        Args:
            save_path: `str`
            validation_set: `bool`.
            validation_split: `float` or None
                float: chunk of `train set` will be marked as `validation set`.
                None: if 'validation set' == True, `validation set` will be
                    copy of `test set`
            shuffle: `str` or None
                None: no any shuffling
                once_prior_train: shuffle train data only once prior train
                every_epoch: shuffle train data prior every epoch
            normalization: `str` or None
                None: no any normalization
                divide_255: divide all pixels by 255
                divide_256: divide all pixels by 256
                by_chanels: substract mean of every chanel and divide each
                    chanel data by it's standart deviation
            one_hot: `bool`, return lasels one hot encoded
        """
        self._save_path = save_path
        self.one_hot = one_hot

        train_fnames, test_fnames = self.get_filenames(self.save_path)

        # add train and validations datasets
        images, labels, block0, block1, block2, logits = read_cifar(train_fnames)
        #block0, block1, block2 = self.read_teacher_output(teacher_output)
        print type(images), images.shape
        if validation_set is not None and validation_split is not None:
            split_idx = int(images.shape[0] * (1 - validation_split))
            self.train = CifarDataSet(
                images=images[:split_idx], labels=labels[:split_idx], block0=block0[:split_idx],
                block1=block1[:split_idx], block2=block2[:split_idx], logits=logits[:split_idx],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation) #block2=block2[split_idx:]
            self.validation = CifarDataSet(
                images=images[split_idx:], labels=labels[split_idx:], block0=block0[split_idx:],
                block1=block1[split_idx:], block2=block2[:split_idx], logits=logits[:split_idx],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation) #block2=block2[split_idx:]
        else:
            self.train = CifarDataSet(
                images=images, labels=labels, block0=block0,
                block1=block1, block2=block2, logits=logits,
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)

        # add test set
        imagest, labelst = self.read_test(test_fnames)
        self.test = CifarDataSet(
            images=imagest, labels=labelst, block0=None,
            block1=None, block2=None, logits=None,
            n_classes=self.n_classes, shuffle=None,
            normalization=normalization,
            augmentation=False)

        if validation_set and not validation_split:
            self.validation = self.test

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = './cifar%d' % self.n_classes #tempfile.gettempdir()
        return self._save_path

    @property
    def data_shape(self):
        return (32, 32, 3)

    @property
    def n_classes(self):
        return self._n_classes

    def get_filenames(self, save_path):
        """Return two lists of train and test filenames for dataset"""
        raise NotImplementedError

    def read_test(self, filename):
        f = h5py.File(filename, 'r')
        images = f['images'][:]
        labels = f['labels'][:]
        f.close()
        #labels_res = self.labels_to_one_hot(labels)        
        return images, labels_res


class Cifar100DataProvider2(CifarDataProvider):
    _n_classes = 100
    data_augmentation = False

    def get_filenames(self, save_path):
        sub_save_path = os.path.join(save_path, 'cifar-100-python')
        train_filenames = 'train_data.h5' #change to your training data file
        test_filenames = 'test_data.h5'  #change to your testing data file
        return train_filenames, test_filenames 

class Cifar10DataProvider2(CifarDataProvider):
    _n_classes = 10
    data_augmentation = False

    def get_filenames(self, save_path):
        sub_save_path = os.path.join(save_path, 'cifar10')
        train_filenames = 'train_data.h5' #change to your training data file
        test_filenames = 'test_data.h5'  #change to your testing data file
        return train_filenames, test_filenames 

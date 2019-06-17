import tempfile
import os
import pickle
import random

import numpy as np
import gzip
from tensorflow.python.platform import gfile
from .base_provider import ImagesDataSet, DataProvider
from .downloader import download_data_url


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def extract_labels(f, num_classes=10):
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    return labels


class MnistDataSet(ImagesDataSet):
    def __init__(self, images, labels, n_classes, shuffle, normalization,
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
            self.shuffle_every_epoch = True
        else:
            raise Exception("Unknown type of shuffling")

        self.images = images
        self.labels = labels
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.normalization = normalization
        self.images = self.normalize_images(images, self.normalization)
        self.start_new_epoch()

    def start_new_epoch(self):
        self._batch_counter = 0
        if self.shuffle_every_epoch:
            images, labels = self.shuffle_images_and_labels(
                self.images, self.labels)
        else:
            images, labels = self.images, self.labels
        if self.augmentation:
            images = augment_all_images(images, pad=4)
        self.epoch_images = images
        self.epoch_labels = labels

    @property
    def num_examples(self):
        return self.labels.shape[0]

    def next_batch(self, batch_size):
        start = self._batch_counter * batch_size
        end = (self._batch_counter + 1) * batch_size
        self._batch_counter += 1
        images_slice = self.epoch_images[start: end]
        labels_slice = self.epoch_labels[start: end]
        if images_slice.shape[0] != batch_size:
            self.start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice


class MNISTDataProvider(DataProvider):
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
        #download_data_url(self.data_url, self.save_path)
        train_data, train_truth, test_data, test_truth = self.get_filenames(self.save_path)

        # add train and validations datasets
        images, labels = self.read_mnist(train_data, train_truth)
        if validation_set is not None and validation_split is not None:
            split_idx = int(images.shape[0] * (1 - validation_split))
            self.train = MnistDataSet(
                images=images[:split_idx], labels=labels[:split_idx],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)
            self.validation = MnistDataSet(
                images=images[split_idx:], labels=labels[split_idx:],
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)
        else:
            self.train = MnistDataSet(
                images=images, labels=labels,
                n_classes=self.n_classes, shuffle=shuffle,
                normalization=normalization,
                augmentation=self.data_augmentation)

        # add test set
        images, labels = self.read_mnist(test_data, test_truth)
        self.test = MnistDataSet(
            images=images, labels=labels,
            shuffle=None, n_classes=self.n_classes,
            normalization=normalization,
            augmentation=False)

        if validation_set and not validation_split:
            self.validation = self.test

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = os.path.join("/home/zhengyu/teacher_student/", 'mnist')#tempfile.gettempdir()
        return self._save_path

    @property
    def data_url(self):
        """Return url for downloaded data depends on cifar class"""
        data_url = ('http://www.cs.toronto.edu/'
                    '~kriz/cifar-%d-python.tar.gz' % self.n_classes)
        return data_url

    @property
    def data_shape(self):
        return (28, 28, 1)

    @property
    def n_classes(self):
        return self._n_classes

    def get_filenames(self, save_path):
        """Return two lists of train and test filenames for dataset"""
        raise NotImplementedError

    def read_mnist(self, data, truth):
    	with gfile.Open(data, 'rb') as f:
            images_res = extract_images(f)
    	with gfile.Open(truth, 'rb') as f:
            labels_res = extract_labels(f)
        if self.one_hot:
            labels_res = self.labels_to_one_hot(labels_res)
        return images_res, labels_res


class MnistDataProvider(MNISTDataProvider):
    _n_classes = 10
    data_augmentation = False

    def get_filenames(self, save_path):
        train_data_filename = os.path.join(save_path, 'train-images-idx3-ubyte.gz')
        train_truth_filename = os.path.join(save_path, 'train-labels-idx1-ubyte.gz')
        test_data_filename = os.path.join(save_path, 't10k-images-idx3-ubyte.gz')
        test_truth_filenames = os.path.join(save_path, 't10k-labels-idx1-ubyte.gz')
        return train_data_filename, train_truth_filename, test_data_filename, test_truth_filenames

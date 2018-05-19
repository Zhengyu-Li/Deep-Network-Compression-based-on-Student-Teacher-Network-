from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
from six.moves import xrange
import tensorflow as tf
import numpy as np
# Parse the input file name
def read_file(filename):

    labels_key = b'fine_labels'

    images_res = []
    labels_res = []
    with open(filename, 'rb') as f:
        images_and_labels = pickle.load(f)#, encoding='bytes'
        images = images_and_labels[b'data']
        images = images.reshape(-1, 3, 32, 32)
        images = images.swapaxes(1, 3).swapaxes(1, 2)
        images_res.append(images)
        labels_res.append(images_and_labels[labels_key])
    images_res = np.vstack(images_res)
    labels_res = np.hstack(labels_res)
    #    if self.one_hot:
    #     labels_res = self.labels_to_one_hot(labels_res)
    return images_res, labels_res

def read_inputs(is_training, args):
    data, l = read_file(args.data_info)
    images = tf.cast(data, tf.float32)
    labels = tf.cast(l, tf.int64)
  # Create a queue that produces the filenames to read.
    if is_training:
        filename_queue = tf.train.slice_input_producer([images, labels], shuffle= args.shuffle, capacity= 1024)
    else:
        filename_queue = tf.train.slice_input_producer([images, labels], shuffle= False,  capacity= 1024, num_epochs =1)

  # Read examples from files in the filename queue.
  #file_content = tf.read_file(filename_queue[0])
  # Read JPEG or PNG or GIF image from file
  #reshaped_image = tf.to_float(tf.image.decode_jpeg(file_content, channels=args.num_channels))
  # Resize image to 256*256
  #reshaped_image = tf.image.resize_images(reshaped_image, args.load_size)

    label = tf.cast(filename_queue[1], tf.int64)
    reshaped_image = filename_queue[0]

    reshaped_image = tf.image.per_image_standardization(reshaped_image)
   # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(50000 * min_fraction_of_examples_in_queue)
  #print(batch_size)
    print ('Filling queue with %d images before starting to train. '
         'This may take some times.' % min_queue_examples)
    batch_size = args.chunked_batch_size if is_training else args.batch_size

  # Load images and labels with additional info 
    images, label_batch = tf.train.batch(
        [reshaped_image, label],
        batch_size= batch_size,
        allow_smaller_final_batch= True if not is_training else False,
        num_threads=args.num_threads,
        capacity=min_queue_examples+3 * batch_size)
    
    return images, label_batch

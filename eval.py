import os
import time
import shutil
from datetime import timedelta

import numpy as np
import tensorflow as tf
from data_providers.utils import get_data_provider_by_name

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logs_path = '/mnt/data/zhengyu/smodel_100_v3/'
save_path = '/mnt/data/zhengyu/'
train_params = {
    'batch_size': 32,
    'iterations of one epoch': 1562,
    'initial_learning_rate': 2e-5,
    'reduce_lr_epoch': 50,  # epochs * 0.5
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': None,  # None, divide_256, divide_255, by_chanels
}

def labels_to_one_hot(labels):
    """Convert 1D array of labels to one hot representation

    Args:
        labels: 1D numpy array
    """
    new_labels = np.zeros((labels.shape[0], 100))
    new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
    return new_labels

def weight_variable_msra(shape, name):
    return tf.get_variable(
        name=name,
        shape=shape,
        initializer=tf.contrib.layers.variance_scaling_initializer())

def weight_variable_xavier(shape, name):
    return tf.get_variable(
        name,
        shape=shape,
        initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape, name='bias'):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name, initializer=initial)

def conv2d(_input, out_features, kernel_size, strides=[1, 1, 1, 1], padding='SAME'):
    in_features = int(_input.get_shape()[-1])
    with tf.name_scope('weights'):
        kernel = weight_variable_msra([kernel_size, kernel_size, in_features, out_features], name='kernel')
        variable_summaries(kernel)
    output = tf.nn.conv2d(_input, kernel, strides, padding)
    return output

def avg_pool(_input, k):
    ksize = [1, k, k, 1]
    strides = [1, k, k, 1]
    padding = 'VALID'
    output = tf.nn.avg_pool(_input, ksize, strides, padding)
    return output

def batch_norm(_input, is_training):
    output = tf.contrib.layers.batch_norm(
        _input, scale=True, is_training=is_training,
        updates_collections=None)
    return output

def dropout(_input, is_training, keep_prob):
    if keep_prob < 1:
        output = tf.cond(
            is_training,
            lambda: tf.nn.dropout(_input, keep_prob),
            lambda: _input
        )
    else:
        output = _input
    return output

def composite_function(_input, out_features, is_training, kernel_size=3, keep_prob=0.5):
    """Function from paper H_l that performs:
    - batch normalization
    - ReLU nonlinearity
    - convolution with required kernel
    - dropout, if required
    """
    with tf.variable_scope("composite_function"):
        # BN
        output = batch_norm(_input, is_training=is_training)
        # ReLU
        output = tf.nn.relu(output)
        # convolution
        output = conv2d(output, out_features=out_features, kernel_size=kernel_size)
        # dropout(in case of training and in case it is no 1.0)
        output = dropout(output, is_training, keep_prob)
    return output

def bottleneck(_input, out_features, is_training, keep_prob=0.5):
    with tf.variable_scope("bottleneck"):
        output = batch_norm(_input, is_training=is_training)
        output = tf.nn.relu(output)
        inter_features = out_features * 4
        output = conv2d(output, out_features=inter_features, kernel_size=1, padding='VALID')
        output = dropout(output, is_training=is_training, keep_prob=keep_prob)
    return output

def add_internal_layer(_input, growth_rate, is_training, bc_mode=True):
    """
    Perform H_l composite function for the layer and after concatenate
    input with output from composite function.
    """
    # call composite function with 3x3 kernel
    comp_out = None
    if not bc_mode:
        comp_out = composite_function(_input, out_features=growth_rate, kernel_size=3, is_training=is_training)
    elif bc_mode:
        bottleneck_out = bottleneck(_input, out_features=growth_rate, is_training=is_training)
        comp_out = composite_function(bottleneck_out, out_features=growth_rate, kernel_size=3, is_training=is_training)
        # concatenate _input with out from composite function
    output = tf.concat(axis=3, values=(_input, comp_out))
    return output

def add_block(_input, growth_rate, layers_per_block, bc_mode, name, is_training):
    """Add N H_l internal layers"""
    output = _input
    for layer in range(layers_per_block):
        with tf.variable_scope("%s_layer_%d" % (name, layer)):
             output = add_internal_layer(output, growth_rate, bc_mode=bc_mode, is_training=is_training)
    return output

def add_block2(_input, growth_rate, layers_per_block, bc_mode, is_training):
    """Add N H_l internal layers"""
    output = _input
    for layer in range(layers_per_block):
        with tf.variable_scope("layer_%d" % layer):
            output = add_internal_layer(output, growth_rate, bc_mode=bc_mode, is_training=is_training)
    return output

def transition_layer(_input, reduction, is_training):
    """Call H_l composite function with 1x1 kernel and after average
    pooling
    """
    # call composite function with 1x1 kernel
    out_features = int(int(_input.get_shape()[-1]) * reduction)
    output = composite_function(_input, out_features=out_features, kernel_size=1, is_training=is_training)
    # run average pooling
    output = avg_pool(output, k=2)
    return output

def transition_layer_to_classes(_input, n_classes, is_training=True, name=None):
    """This is last transition to get probabilities by classes. It perform:
    - batch normalization
    - ReLU nonlinearity
    - wide average pooling
    - FC layer multiplication
    """
    # BN
    output = batch_norm(_input, is_training=is_training)
    # ReLU
    output = tf.nn.relu(output)
    # average pooling
    last_pool_kernel = int(output.get_shape()[-2])
    output = avg_pool(output, k=last_pool_kernel)
    # FC
    features_total = int(output.get_shape()[-1])
    output = tf.reshape(output, [-1, features_total])
    with tf.name_scope('weights2'):
        W = weight_variable_xavier([features_total, n_classes], name=name)
        variable_summaries(W)
    with tf.name_scope('bias'):
        bias = bias_variable([n_classes])
        variable_summaries(bias)

    logits = tf.matmul(output, W) + bias
    return logits


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))

def build_student(images,
                  is_training,
                  depth,
                  blocks=3,
                  bc_mode=True,
                  n_classes=None):

    growth_rate = 12
    layers_per_block = (depth-blocks-1)//blocks
    # first - initial 3 x 3 conv to first_output_features
    if not bc_mode:
        print("Build Student %s model with %d blocks, "
              "%d composite layers each." % (
                  'DenseNet', blocks, layers_per_block))
    if bc_mode:
        layers_per_block = layers_per_block // 2
        print("Build Student %s model with %d blocks, "
              "%d bottleneck layers and %d composite layers each." % (
                  'DenseNet-BC', blocks, layers_per_block, layers_per_block))

    #with tf.name_scope("Student"):
    with tf.name_scope('Initial_convolution'):
        with tf.variable_scope("Student_Initial_convolution"):
            output = conv2d(images, out_features=growth_rate * 2, kernel_size=3)

    # add N required blocks
    with tf.name_scope('Block_0'):
        with tf.variable_scope("Student_Block_0"):
            block0_out = add_block(output, growth_rate, layers_per_block, bc_mode, name='block0', is_training=is_training)
    #        block0_out = composite_function(block0_out, out_features=1320, kernel_size=1, is_training=is_training)

        # output feature = 48
    with tf.name_scope('Transition_after_Block_0'):
        with tf.variable_scope("Student_Transition_after_Block_0"):
            #tran0 = transition_layer(block0_out, 0.5, is_training)
            block0_output = composite_function(block0_out, out_features=660, kernel_size=1, is_training=is_training)
            block0_output = avg_pool(block0_output, k=2)

    with tf.name_scope('Block_1'):
        with tf.variable_scope("Student_Block_1"):
            block1_out = add_block(block0_output, growth_rate, layers_per_block, bc_mode, name='block1', is_training=is_training)
     #       block1_out = composite_function(block1_out, out_features=1900, kernel_size=1, is_training=is_training)

    with tf.name_scope('Transition_after_Block_1'):
        with tf.variable_scope("Student_Transition_after_Block_1"):
            #tran1 = transition_layer(block1_out, 0.5, is_training)
            block1_output = composite_function(block1_out, out_features=950, kernel_size=1, is_training=is_training)
            block1_output = avg_pool(block1_output, k=2)

    with tf.name_scope('Block_2'):
        with tf.variable_scope("Student_Block_2"):
            block2_output = add_block(block1_output, growth_rate, layers_per_block, bc_mode, name='block2', is_training=is_training)
      #      block2_output = composite_function(block2_output, out_features=2190, kernel_size=1, is_training=is_training)

    with tf.name_scope('FC'):
            with tf.variable_scope("Student_Transition_to_classes"):
                #to_1024 = composite_function(block2_output, out_features=1024, kernel_size=1, is_training=is_training)
                logits = transition_layer_to_classes(block2_output, n_classes, is_training, name='W')

    return block0_output, block1_output, logits

def my_gradient(g0, g1, g3):

    g0_copy = []
    g1_copy = []
    g3_copy = []
    for i in range(0, len(g3)):

        (grad, val) = g3[i]

        if val is not None:
            if i < len(g0):
                g3_copy.append((g3[i][0]+g1[i][0]+g0[i][0])/3)

            elif i >= len(g0) and i < len(g1):
                g3_copy.append((g3[i][0]+g1[i][0])/2)

            #elif i >= len(g1) and i < len(g2):
            #    g3_copy.append((g3[i][0]+g2[i][0])/2)

            else:
                g3_copy.append(g3[i][0])
        else:
            print ("failure mode : no grads")
            continue
    return g3_copy


data_provider = get_data_provider_by_name('C100', train_params)
data_shape = data_provider.data_shape
n_classes = 100
data = data_provider.train
student_depth = 100
nesterov_momentum = 0.9
batch_size = 32
weight_decay = 1e-4
lr = float(1e-6)
val = data_provider.test

if __name__ == '__main__':

    shape = [None]
    shape.extend(data_shape)
    with tf.name_scope('input'):
        student_input = tf.placeholder(tf.float32, shape=shape, name='input_student')
    with tf.name_scope('learning_rate0'):
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    with tf.name_scope('teacher_output0'):
        teacher_output0 = tf.placeholder(tf.float32, shape=[None, 16, 16, 660], name='input_student')
    with tf.name_scope('teacher_output1'):
        teacher_output1 = tf.placeholder(tf.float32, shape=[None, 8, 8, 950], name='input_student')
    with tf.name_scope('teacher_logits'):
        teacher_logits = tf.placeholder(tf.float32, shape=[None, 100], name='input_student')
    with tf.name_scope('labels'):
        labels = tf.placeholder(tf.float32, shape=[None, n_classes], name='labels')
    with tf.name_scope('is_training'):
        is_training = tf.placeholder(tf.bool, shape=[])

    student_output0, student_output1, student_logits = build_student(images=student_input,
                                                                      is_training=is_training,
                                                                      n_classes=n_classes,
                                                                      depth=student_depth)
    student_pred = tf.nn.softmax(student_logits)
    teacher_pred = tf.nn.softmax(teacher_logits)
    with tf.name_scope('optimizer'):
        optimizer = tf.train.MomentumOptimizer(learning_rate, nesterov_momentum, use_nesterov=True)

    with tf.name_scope("l2_0"):
        l2_0 = tf.reduce_mean(tf.nn.l2_loss(student_output0-teacher_output0))
    with tf.name_scope("l2_1"):
        l2_1 = tf.reduce_mean(tf.nn.l2_loss(student_output1-teacher_output1))
    with tf.name_scope("l2_2"):
        l2_2 = tf.reduce_mean(tf.nn.l2_loss(student_logits-teacher_logits))
    tf.summary.scalar('l2_0', l2_0)
    tf.summary.scalar('l2_1', l2_1)
    tf.summary.scalar('l2_2', l2_2)


    gradient_all = optimizer.compute_gradients(l2_2)
    init = [v for (g, v) in gradient_all if 'Student_Initial' in v.name]
    grads_vars0 = [v for (g, v) in gradient_all if 'Block_0' in v.name]
    grads_vars1 = [v for (g, v) in gradient_all if 'Block_1' in v.name]
    grads_vars2 = [v for (g, v) in gradient_all if 'Block_2' in v.name]
    trans = [v for (g, v) in gradient_all if 'Student_Transition_to_classes' in v.name]

    with tf.name_scope("l2_loss0"):
        l2_loss0 = tf.add_n([tf.nn.l2_loss(var) for var in init+grads_vars0])
    with tf.name_scope("l2_loss1"):
        l2_loss1 = tf.add_n([tf.nn.l2_loss(var) for var in init+grads_vars0+grads_vars1])
    with tf.name_scope("l2_loss2"):
        l2_loss2 = tf.add_n([tf.nn.l2_loss(var) for var in init+grads_vars0+grads_vars1+grads_vars2+trans])
    tf.summary.scalar('l2_loss0', l2_loss0)
    tf.summary.scalar('l2_loss1', l2_loss1)
    tf.summary.scalar('l2_loss2', l2_loss2)

    with tf.name_scope("loss0"):
        loss0 = l2_0+l2_loss0
    with tf.name_scope("loss1"):
        loss1 = l2_1+l2_loss1
    with tf.name_scope("loss2"):
        loss2 = l2_2+l2_loss2
    tf.summary.scalar('loss0', loss0)
    tf.summary.scalar('loss1', loss1)
    tf.summary.scalar('loss2', loss2)

    with tf.name_scope('gradient0'):
        gradient0 = optimizer.compute_gradients(loss0, init+grads_vars0)
    with tf.name_scope('gradient1'):
        gradient1 = optimizer.compute_gradients(loss1, init+grads_vars0+grads_vars1)
    with tf.name_scope('gradient2'):
        gradient2 = optimizer.compute_gradients(loss2, init+grads_vars0+grads_vars1+grads_vars2+trans)


    grads_holder = [(tf.placeholder(tf.float32, shape=g.get_shape()), v) for (g, v) in gradient2]

    with tf.name_scope('apply_gradient2'):
        train_op = optimizer.apply_gradients(grads_holder)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(student_pred, 1), tf.argmax(labels, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('t_accuracy'):
        with tf.name_scope('t_correct_prediction'):
            t_correct_prediction = tf.equal(tf.argmax(teacher_pred, 1), tf.argmax(labels, 1))
        with tf.name_scope('accuracy'):
            t_accuracy = tf.reduce_mean(tf.cast(t_correct_prediction, tf.float32))
    tf.summary.scalar('t_accuracy', t_accuracy)

    merged = tf.summary.merge_all()
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.70)
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=5)
    logswriter = tf.summary.FileWriter(logs_path)
    logswriter.add_graph(sess.graph)
    summary_writer = logswriter
    global_step = 0
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state('/mnt/data/zhengyu/smodel_100_v3/')
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loading success, global_step is %s' % global_step)
    else:
        print('No checkpoint file found, global_step is %s' % global_step)

    total_start_time = time.time()

    accuracy_v = []

    for i in range(10000 // 200):
        batch = val.next_batch(200,False)
        feed_dict_v = {
            student_input: batch[0],
            labels: batch[1],
            is_training: False,
        }
        val_accuracy = sess.run(accuracy, feed_dict=feed_dict_v)
        accuracy_v.append(val_accuracy)
        print (i, val_accuracy)

    mean_accuracy_v = np.mean(np.asarray(accuracy_v))

    print ("validation accuracy: %f" % mean_accuracy_v)

    summary = tf.Summary(value=[
        tf.Summary.Value(
            tag='accuracy_s_val', simple_value=float(mean_accuracy_v))
    ])
    logswriter.add_summary(summary, epoch)


    total_training_time = time.time() - total_start_time
    print("\nTotal testing time: %s" % str(timedelta(seconds=total_training_time)))

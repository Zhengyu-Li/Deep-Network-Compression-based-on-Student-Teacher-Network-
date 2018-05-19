import tensorflow as tf
import pickle
import h5py
import cPickle
import numpy as np
import os
from data_providers.utils import get_data_provider_by_name
n_classes = 100

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

train_params_cifar = {
    'batch_size': 32,
    'iterations of one epoch': 1562,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,  # epochs * 0.5
    'reduce_lr_epoch_2': 225,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
}

def labels_to_one_hot(labels):
    """Convert 1D array of labels to one hot representation

    Args:
        labels: 1D numpy array
    """
    new_labels = np.zeros((labels.shape[0], n_classes))
    new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
    return new_labels

def read_cifar(filenames):
    labels_key = b'fine_labels'

    images_res = []
    labels_res = []
    #for fname in filenames:
    with open(filenames, 'rb') as f:
        images_and_labels = pickle.load(f)  # , encoding='bytes'
    images = images_and_labels[b'data']
    images = images.reshape(-1, 3, 32, 32)
    images = images.swapaxes(1, 3).swapaxes(1, 2)
    images_res.append(images)
    labels_res.append(images_and_labels[labels_key])
    images_res = np.vstack(images_res)
    labels_res = np.hstack(labels_res)
    labels_res = labels_to_one_hot(labels_res)
    #print labels_res[1]
    return images_res, labels_res

def measure_mean_and_std(images):
    # for every channel in image
    means = []
    stds = []
    # for every channel in image(assume this is last dimension)
    for ch in range(images.shape[-1]):
        means.append(np.mean(images[:, :, :, ch]))
        stds.append(np.std(images[:, :, :, ch]))
    return means, stds

def normalize_images(images, normalization_type):
    """
    Args:
        images: numpy 4D array
        normalization_type: `str`, available choices:
            - divide_255
            - divide_256
            - by_chanels
    """
    if normalization_type == 'divide_255':
        images = images / 255
    elif normalization_type == 'divide_256':
        images = images / 256
    elif normalization_type == 'by_chanels':
        images = images.astype('float64')
        images_means, images_stds = measure_mean_and_std(images)
        # for every channel in image(assume this is last dimension)
        for i in range(images.shape[-1]):
            images[:, :, :, i] = ((images[:, :, :, i] - images_means[i]) /
                                  images_stds[i])
    else:
        raise Exception("Unknown type of normalization")
    return images

TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))

class DenseNet:
    def __init__(self, growth_rate=12, depth=100,
                 total_blocks=3, keep_prob=0.8,
                 weight_decay=1e-4, nesterov_momentum=0.9, model_type='DenseNet-BC', dataset='C100',
                 reduction=0.5,
                 bc_mode=True,
                 **kwargs):
        """
        Class to implement networks from this paper
        https://arxiv.org/pdf/1611.05552.pdf

        Args:
            data_provider: Class, that have all required data sets
            growth_rate: `int`, variable from paper
            depth: `int`, variable from paper
            total_blocks: `int`, paper value == 3
            keep_prob: `float`, keep probability for dropout. If keep_prob = 1
                dropout will be disables
            weight_decay: `float`, weight decay for L2 loss, paper = 1e-4
            nesterov_momentum: `float`, momentum for Nesterov optimizer
            model_type: `str`, 'DenseNet' or 'DenseNet-BC'. Should model use
                bottle neck connections or not.
            dataset: `str`, dataset name
            should_save_logs: `bool`, should logs be saved or not
            should_save_model: `bool`, should model be saved or not
            renew_logs: `bool`, remove previous logs for current model
            reduction: `float`, reduction Theta at transition layer for
                DenseNets with bottleneck layers. See paragraph 'Compression'
                https://arxiv.org/pdf/1608.06993v3.pdf#4
            bc_mode: `bool`, should we use bottleneck layers and features
                reduction or not.
        """
      
        self.data_shape = (32, 32, 3)
        self.n_classes = 100
        self.depth = depth
        self.growth_rate = growth_rate
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.bc_mode = bc_mode
        # compression rate at the transition layers
        self.reduction = reduction
        if not bc_mode:
            print("Build %s model with %d blocks, "
                  "%d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block))
        if bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print("Build %s model with %d blocks, "
                  "%d bottleneck layers and %d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block,
                      self.layers_per_block))
        print("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        self.dataset_name = dataset
        self.batches_step = 0

        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(gpu_options = gpu_options)
        # restrict model GPU memory utilization to min required
        #config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'saves/%s' % self.model_identifier
            if not os.path.exists(save_path):
                os.makedirs(save_path)  # , exist_ok=True)
            save_path = os.path.join(save_path, 'model.ckpt')
            self._save_path = save_path
        return save_path

    @property
    def model_identifier(self):
        return "{}_growth_rate={}_depth={}_dataset_{}".format(
            self.model_type, self.growth_rate, self.depth, self.dataset_name)

    def load_model(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        t = 0
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            t = int(global_step)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('Loading success, global_step is %s' % global_step)
        else:
            print ("false to load model")

        return t


    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_images')
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels')
        self.learning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[])

    def composite_function(self, _input, out_features, kernel_size=3):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
        return output

    def bottleneck(self, _input, out_features):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if not self.bc_mode:
            comp_out = self.composite_function(
                _input, out_features=growth_rate, kernel_size=3)
        elif self.bc_mode:
            bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
            comp_out = self.composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3)
        # concatenate _input with out from composite function
        if TF_VERSION >= 1.0:
            output = tf.concat(axis=3, values=(_input, comp_out))
        else:
            output = tf.concat(3, (_input, comp_out))
        return output

    def add_block(self, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output_c = self.composite_function(
            _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = self.avg_pool(output_c, k=2)
        return output

    def transition_layer_to_classes(self, _input):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W = self.weight_variable_xavier(
            [features_total, self.n_classes], name='W')
        bias = self.bias_variable([self.n_classes])
        logits = tf.matmul(output, W) + bias
        return logits

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_training,
            updates_collections=None)
        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def _build_graph(self):
    '''
    Make sure the structure and name are same as in pretrained model.
    You can choose other intermediate outputs from teacher for guiding the student.
    '''
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        # first - initial 3 x 3 conv to first_output_features
        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                self.images,
                out_features=self.first_output_features,
                kernel_size=3)

        # name of variable must be the same as it in the teacher network
        with tf.variable_scope("Block_0"):
            output0 = self.add_block(output, growth_rate, layers_per_block)
        with tf.variable_scope("Transition_after_block_0"):
            t_output0 = self.transition_layer(output0)

        with tf.variable_scope("Block_1"):
            output1 = self.add_block(t_output0, growth_rate, layers_per_block)
        with tf.variable_scope("Transition_after_block_1"):
            t_output1 = self.transition_layer(output1)

        with tf.variable_scope("Block_2"):
            output2 = self.add_block(t_output1, growth_rate, layers_per_block)

        with tf.variable_scope("Transition_to_classes"):
            logits = self.transition_layer_to_classes(output2)

        prediction = tf.nn.softmax(logits)
        # choose intermediate outputs
        self.blockoutput0 = t_output0 #optional
        self.blockoutput1 = t_output1 #optional
        self.blockoutput2 = output2 #optional
        self.out = logits #optional
        # Losses
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels))
        self.cross_entropy = cross_entropy
        l2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # optimizer and train step
        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.train_step = optimizer.minimize(
            cross_entropy + l2_loss * self.weight_decay)

        correct_prediction = tf.equal(
            tf.argmax(prediction, 1),
            tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def test(self, images, labels):
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        # configration=tf.ConfigProto(gpu_options=gpu_options)
        feed_dict = {
                self.images: images,
                self.labels: labels,
                self.is_training: False,    
        }
        fetches = [self.blockoutput0, self.blockoutput1, self.blockoutput2, self.out]
        out0, out1, out2, out3 = self.sess.run(fetches, feed_dict=feed_dict)
        return out0, out1, out2, out3

if __name__ == '__main__':
    img, labels = read_cifar('./teacher_student/cifar100/cifar-100-python/train') #cifar-100 training data path
    img = normalize_images(img, 'by_chanels')
    model = DenseNet()
    model_path = './teacher_student/saves/' #pretrained model folder
    model.load_model(model_path)
    print ("save file...")
    
    block0, block1, block2, logit = model.test(img[0:2500], labels[0:2500])
    for i in range(2500, 50000, 2500):
        out0, out1, out2, log = model.test(img[i:i+2500], labels[i:i+2500])
        block0 = np.concatenate((block0, out0)) #optional
        block1 = np.concatenate((block1, out1)) #optional
        block2 = np.concatenate((block2, out2)) #optional
        logit = np.concatenate((logit, log)) #optional
        print block0.shape, block1.shape, block2.shape, logit.shape

    f = h5py.File('./train_data.h5', 'w') # saved file path
    f['images'] = img
    f['labels'] = labels
    f['block0'] = block0 #optional
    f['block1'] = block1 #optional
    f['block2'] = block2 #optional
    f['logits'] = logit #optional
    f.close()
    print ("finished.")

    test_img, test_labels = read_cifar('./teacher_student/cifar100/cifar-100-python/test') #cifar-100 training data path
    test_img = normalize_images(test_img, 'by_chanels')
    f = h5py.File('./test_data.h5', 'w') # saved file path
    f['images'] = test_img
    f['labels'] = test_labels
    f.close()
    print ("finished.")   
    




    

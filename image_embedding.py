import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


class ImageEmbedding(object):
    def __init__(self, hidden_num, channels, conv_initializer, k=5):
        self.hidden_num = hidden_num
        self.channels = channels
        with tf.variable_scope('image_embedding', reuse=tf.AUTO_REUSE):
            self.conv1_kernel = tf.get_variable('conv1_kernel', [k, k, self.channels, self.hidden_num],
                                                initializer=conv_initializer)

            self.conv2_kernel = tf.get_variable('conv2_kernel', [k, k, self.hidden_num, self.hidden_num],
                                                initializer=conv_initializer)
        self.activation = tf.nn.relu

    def model(self, images):
        conv = tf.nn.conv2d(images, self.conv1_kernel, [1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(conv, name='conv1_post_activation')

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1')

        conv2 = tf.nn.conv2d(norm1, self.conv2_kernel, [1, 1, 1, 1], padding='SAME')
        conv2_act = tf.nn.relu(conv2, name='conv2_post_activation')

        norm2 = tf.nn.lrn(conv2_act, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        with tf.variable_scope('local3', reuse=tf.AUTO_REUSE):
            image_reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
            dim = image_reshape.get_shape()[1].value
            local3_weight = tf.get_variable(name='weight', shape=[dim, 384],
                                            initializer=tf.truncated_normal_initializer(stddev=0.04))
            local3_biases = tf.get_variable(name='biases', shape=[384], initializer=tf.constant_initializer(0.1))
            local3=tf.nn.relu(tf.matmul(image_reshape, local3_weight)+local3_biases, name='local3_dense')

        with tf.variable_scope('local4', reuse=tf.AUTO_REUSE) as scope:
            local4_weight = tf.get_variable(name='weight', shape=[384, 64],
                                            initializer=tf.truncated_normal_initializer(stddev=0.04))
            local4_biases = tf.get_variable(name='biases', shape=[64], initializer=tf.constant_initializer(0.1))
            local4 = tf.nn.relu(tf.matmul(local3, local4_weight) + local4_biases, name='local4_dense')
        return local4

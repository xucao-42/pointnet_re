import tensorflow as tf


class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum=0.99, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, phase):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            epsilon=self.epsilon,
                                            updates_collections=None,
                                            scale=True,
                                            is_training=phase,
                                            scope=self.name)


def dense_bn_nonlinear(inputs, units,
                       use_bn=True,
                       phase=None,
                       activation_fn=tf.nn.relu,
                       scope="fc"):
    """

    :param inputs: tensor of shape (batch_size, ...,n) from last layer
    :param units: output units
    :param use_bn: a boolean value indicating whether using bn before activation function or not
    :param phase: a boolean placeholder of shape () indicating whether its in training phase or test phase
    :param activation_fn:
    :param scope: scope name
    :return: (batch_size, ...,units)
    """
    with tf.variable_scope(scope):
        out = tf.layers.dense(inputs, units,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        if use_bn:
            batch_norm = BatchNorm()
            if phase is None:
                raise ValueError("Phase is not given!")
            out = batch_norm(out, phase=phase)
        if activation_fn is not None:
            out = activation_fn(out)
        return out

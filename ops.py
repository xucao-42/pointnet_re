import tensorflow as tf


class BatchNorm(object):
    def __init__(self, epsilon=1e-5, momentum=0.99, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, is_training):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            epsilon=self.epsilon,
                                            updates_collections=None,
                                            scale=True,
                                            center=True,
                                            is_training=is_training,
                                            scope=self.name)


def dense_norm_nonlinear(inputs, units,
                       norm_type=None,
                       is_training=None,
                       activation_fn=tf.nn.relu,
                       scope="fc"):
    """

    :param inputs: tensor of shape (batch_size, ...,n) from last layer
    :param units: output units
    :param norm_type: a string indicating which type of normalization is used.
                    A string start with "b": use batch norm.
                    A string starting with "l": use layer norm
                    others: do not use normalization
    :param is_training: a boolean placeholder of shape () indicating whether its in training phase or test phase.
    It is only needed when BN is used.
    :param activation_fn:
    :param scope: scope name
    :return: (batch_size, ...,units)
    """
    with tf.variable_scope(scope):
        out = tf.layers.dense(inputs, units,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
        #  batch_size, num_point, num_features
        tf.contrib.layers.layer_norm(out, )
        if norm_type is not None:
            if norm_type.lower().startswith("b"):
                batch_norm = BatchNorm()
                if is_training is None:
                    raise ValueError("is_training is not given!")
                out = batch_norm(out, is_training=is_training)
            elif norm_type.lower().startswith("l"):
                out = tf.contrib.layers.layer_norm(out, scope="layer_norm")
            else:
                raise ValueError("please give the right norm type beginning with 'b' or 'l'!")
        if activation_fn is not None:
            out = activation_fn(out)
        return out

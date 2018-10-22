from utils import *
from ops import *


def get_input_placeholders(batch_size, num_point, num_feature):
    pts_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_feature), name="point_clouds")
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size,), name="labels")
    is_training_pl = tf.placeholder(tf.bool, shape=(), name="phase")
    keepprob_pl = tf.placeholder(tf.float32, shape=(), name="keep_prob")
    return pts_pl, labels_pl, is_training_pl, keepprob_pl


def input_transform_net(pts, is_training_pl, norm_type, scope):
    """
    :param pts: input point clouds of shape (batch_size, num_pts, k)
    :param is_training_pl: a boolean placeholder of shape () indicating whether its in training phase or testing phase
    :param norm_type: a string indicating which type of normalization is used.
                    A string start with "b": use batch norm.
                    A string starting with "l": use layer norm
                    others: do not use normalization
    :param scope:
    :return: transform matrix of shape (3,3)
    """
    batch_size = pts.get_shape()[0].value
    k = pts.get_shape()[-1].value
    mlp_list = [64, 128, 1024]
    global_mlp_list = [512, 256]
    with tf.variable_scope(scope):
        net = pts
        for idx, units in enumerate(mlp_list):
            net = dense_norm_nonlinear(net, units, norm_type=norm_type, is_training=is_training_pl,scope="fc{}".format(idx + 1))
        net = tf.reduce_max(net, axis=1, keepdims=False)  # (batch_size, 1024)
        for idx, units in enumerate(global_mlp_list):
            net = dense_norm_nonlinear(net, units, norm_type="bn", is_training=is_training_pl, scope="global_feature_fc{}".format(idx + 1))
        
        net = tf.layers.dense(net, units=k*k, activation=None,
                              kernel_initializer=tf.constant_initializer(0.0),
                              bias_initializer=tf.constant_initializer(np.eye(k).flatten()),
                              name="final_fc")
        transform_matrix = tf.reshape(net, [batch_size, k, k])
    return transform_matrix


def get_model(pts, keep_prob, is_training_pl, norm_type, num_label):
    """
    the classification network. Input is point clouds of shape (batch_size, num_pts, num_features). Output is
    point cloud-wise logits of shape (batch_size, num_classes).

    :param pts:
    :param keep_prob:
    :param phase:
    :param use_bn:
    :param num_label:
    :return:
    """
    transform_matrices = []
    transform_matrix1 = input_transform_net(pts, is_training_pl, norm_type, scope="T_Net1")
    transform_matrices.append(transform_matrix1)
    pts_transformed = tf.matmul(pts, transform_matrix1, name="transform1")

    net = dense_norm_nonlinear(pts_transformed, 64, norm_type=norm_type,
                               is_training=is_training_pl, scope="fc01")
    net = dense_norm_nonlinear(net, 64, norm_type=norm_type,
                               is_training=is_training_pl, scope="fc02")

    transform_matrix2 = input_transform_net(net, is_training_pl, norm_type, scope="T_Net2")
    transform_matrices.append(transform_matrix2)
    net = tf.matmul(net, transform_matrix2, name="transform2")

    mlp_list = [64, 128, 1024]
    global_mlp_list = [512, 256]
    
    for idx, units in enumerate(mlp_list):
        net = dense_norm_nonlinear(net, units, norm_type=norm_type,
                                   is_training=is_training_pl, scope="fc{}".format(idx + 1))

    net = tf.reduce_max(net, axis=1, keepdims=False)  # (batch_size, 1024)

    for idx, units in enumerate(global_mlp_list):
        net = dense_norm_nonlinear(net, units, norm_type="bn",
                                   is_training=is_training_pl, scope="global_fc{}".format(idx + 1))
        net = tf.nn.dropout(net, keep_prob, name="dropout{}".format(idx + 1))

    logits = dense_norm_nonlinear(net, num_label, norm_type=None, activation_fn=None, scope="final_fc")

    return logits, transform_matrices


def get_loss(logits, labels, transform_matrices, reg_weight=0.001):
    """

    :param logits: of shape (batch_size, num_classes)
    :param labels:  of shape (batch_size, )
    :param transform_matrices: a tensor
    :param reg_weight: regularization factor of transform matrix
    :return:
    """
    classify_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    tf.summary.scalar('classify loss', classify_loss)

    #  Enforce the transformation as orthogonal matrix
    mat_diff_loss = []
    for transform_matrix in transform_matrices:
        k = transform_matrix.get_shape()[1].value
        mat_diff = tf.matmul(transform_matrix, tf.transpose(transform_matrix, perm=[0, 2, 1]))
        mat_diff -= tf.constant(np.eye(k), dtype=tf.float32)
        mat_diff_loss.append(tf.nn.l2_loss(mat_diff))
    mat_diff_loss = tf.add_n(mat_diff_loss)
    tf.summary.scalar('mat loss', mat_diff_loss)
    total_loss = classify_loss + mat_diff_loss * reg_weight
    tf.summary.scalar('total loss', total_loss)
    return total_loss, classify_loss, mat_diff_loss


def get_batch_acc(logits, labels):
    batch_size = logits.get_shape()[0].value
    correct = tf.equal(tf.argmax(logits, -1), tf.to_int64(labels))
    acc = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(batch_size)
    tf.summary.scalar('batch_accuracy', acc)
    return acc


if __name__ == "__main__":
    batch_size = 32
    num_point = 2048
    num_feature = 3
    num_label = 40
    pts_pl, labels_pl, is_training_pl, dropout_pl = get_input_placeholders(batch_size, num_point, num_feature)
    logits, transform_matrices = get_model(pts_pl, dropout_pl, is_training_pl, norm_type="bn", num_label=40)
    total_loss, classify_loss, mat_diff_loss = get_loss(logits, labels_pl, transform_matrices)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_val = sess.run([total_loss, classify_loss, mat_diff_loss],
                            feed_dict={pts_pl: np.random.randn(batch_size, num_point, num_feature),
                                       labels_pl: np.random.randint(num_label, size=(batch_size,)),
                                       is_training_pl: True,
                                       dropout_pl: 0.7})
        print(loss_val)  # should be around -log(1/num_label)

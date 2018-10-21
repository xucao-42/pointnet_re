import socket, os, configparser, time, shutil
from pointnet_cls import *
import provider
import tensorflow as tf
np.random.seed(42)
# download modelnet40
HOSTNAME = socket.gethostname()
DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))

# read hyperparameters
config = configparser.ConfigParser()
config.read("base_config.ini")
BATCH_SIZE = config["hyperparameters"].getint("batch_size")
NUM_POINT = config["hyperparameters"].getint("num_pts")
EPOCH = config["hyperparameters"].getint("epoch")
BASE_LEARNING_RATE = config["hyperparameters"].getfloat("lr")
DECAY_STEP = config["hyperparameters"].getint("lr_decay_step")
DECAY_RATE = config["hyperparameters"].getfloat("lr_decay_rate")


def log_string(out_str, log_fout):
    log_fout.write(out_str+'\n')
    log_fout.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


# ModelNet40 official train/test split
TRAIN_FILES = getDataFiles(
    os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = getDataFiles(
    os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048/test_files.txt'))


def train(train_save_dir, log_file):
    # directory for saving the intermediate model
    model_dir = os.path.join(train_save_dir, "model")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    batch = tf.Variable(0)
    learning_rate = get_learning_rate(batch)
    tf.summary.scalar('learning_rate', learning_rate)

    pts_pl, labels_pl, is_training_pl, keepprob_pl = get_input_placeholders(BATCH_SIZE, NUM_POINT, 3)
    logits_ts, transform_matrices_ts = get_model(pts_pl, keepprob_pl, is_training_pl, use_bn=True, num_label=40)
    total_loss_ts, classify_loss_ts, mat_diff_loss_ts = get_loss(logits_ts, labels_pl, transform_matrices_ts)
    batch_acc = get_batch_acc(logits_ts, labels_pl)

    optim_op = tf.train.AdamOptimizer(learning_rate, name="optim_op").minimize(total_loss_ts, global_step=batch)
    saver = tf.train.Saver(max_to_keep=200)
    merged_summary = tf.summary.merge_all()

    def train_one_epoch():
        # Shuffle training files to vary the order of training files(hdf5) at each epoch
        train_file_idxs = np.arange(0, len(TRAIN_FILES))
        np.random.shuffle(train_file_idxs)

        for fn in range(len(TRAIN_FILES)):
            log_string('----' + str(fn) + '-----', log_file)
            current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
            current_data = current_data[:, 0:NUM_POINT, :]
            current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
            current_label = np.squeeze(current_label)

            total_num_samples = current_data.shape[0]
            num_batches = total_num_samples // BATCH_SIZE
            total_correct = 0
            total_seen = 0
            all_total_loss = 0
            all_classify_loss =0

            for batch_idx in range(num_batches):
                total_seen += BATCH_SIZE
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx + 1) * BATCH_SIZE
                rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx])
                jittered_data = provider.jitter_point_cloud(rotated_data)
                current_batch_data = jittered_data
                current_batch_label = current_label[start_idx:end_idx]
                # display_point(current_batch_data[0], 127 * np.ones_like(current_batch_data[0]))
                _, step, total_loss_train, classify_loss_train, mat_diff_loss_train, summary_train, logits_train = \
                    sess.run([optim_op, batch, total_loss_ts, classify_loss_ts, mat_diff_loss_ts, merged_summary, logits_ts],
                             feed_dict={pts_pl: current_batch_data,
                                        labels_pl: current_batch_label,
                                        is_training_pl: True,
                                        keepprob_pl: 0.7})
                train_writer.add_summary(summary_train, step)
                all_total_loss += total_loss_train
                all_classify_loss += classify_loss_train
                total_correct += np.sum(np.argmax(logits_train, 1) == current_batch_label)

            log_string('mean total loss: {:.4f}'.format(all_total_loss / float(num_batches)), log_file)
            log_string('mean classify loss: {:.4f}'.format(all_classify_loss / float(num_batches)), log_file)
            log_string('accuracy: {:.4f}'.format(total_correct / float(total_seen)), log_file)

    def eval_one_epoch():
        total_correct = 0
        total_seen = 0
        all_total_loss = 0
        all_classify_loss = 0
        total_seen_class = [0 for _ in range(40)]
        total_correct_class = [0 for _ in range(40)]

        for fn in range(len(TEST_FILES)):
            log_string('----' + str(fn) + '-----', log_file)
            current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
            current_data = current_data[:, 0:NUM_POINT, :]
            current_label = np.squeeze(current_label)
            file_size = current_data.shape[0]
            num_batches = file_size // BATCH_SIZE
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = (batch_idx + 1) * BATCH_SIZE
                current_batch_data = current_data[start_idx:end_idx]
                current_batch_label = current_label[start_idx:end_idx]
                total_loss_val, classify_loss_val, logits_val = \
                    sess.run([total_loss_ts, classify_loss_ts, logits_ts],
                             feed_dict={pts_pl: current_batch_data,
                                        labels_pl: current_batch_label,
                                        is_training_pl: True,
                                        keepprob_pl: 1.0})
                pred_val = np.argmax(logits_val, 1)
                total_correct += np.sum(pred_val == current_batch_label)
                total_seen += BATCH_SIZE
                all_total_loss += total_loss_val * BATCH_SIZE
                all_classify_loss += classify_loss_val * BATCH_SIZE
                for i in range(start_idx, end_idx):
                    l = current_label[i]
                    total_seen_class[l] += 1
                    total_correct_class[l] += (pred_val[i - start_idx] == l)
        log_string('eval mean total loss: {:.4f}'.format(all_total_loss / total_seen), log_file)
        log_string('eval mean classify loss: {:.4f}'.format(all_classify_loss / total_seen), log_file)
        log_string('eval overall  accuracy: {:.4f}'.format(total_correct / float(total_seen)), log_file)
        log_string('eval avg class accuracy: {:.4f}'.format
                   (np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))), log_file)

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(train_dir, sess.graph)  # save model graph
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCH):
            log_string('**** EPOCH {:3d} ****'.format(i), log_file)
            train_one_epoch()
            if i % 3 == 0:
                saver.save(sess, model_dir + "/model", i)
            eval_one_epoch()


if __name__ == "__main__":
    if not os.path.exists("train_results"):
        os.mkdir("train_results")
    train_time = str(time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time())))
    train_dir = os.path.join("train_results", train_time)
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    LOG_FOUT = open(os.path.join(train_dir, 'log.txt'), 'w')
    shutil.copyfile("base_config.ini", os.path.join(train_dir, "config.ini"))
    train(train_dir, LOG_FOUT)

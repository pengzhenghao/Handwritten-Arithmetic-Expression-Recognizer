# 本程序训练网络。
import os
import time

import cv2
import numpy as np
import tensorflow as tf

starttime = time.time()
tmptime = starttime
sess = tf.InteractiveSession()


# log_dir = './log/526test/526_8'
# 526 use 64 fc
# 526_2 use 256 fc
# 526_3 use 400 fc
# 526_4 use 400 fc and 2nd cnn shape is [3,3] (above are [1,1])
# 526_5 use 400fc and 3 cnn layer with 2nd shape[3,3]
# 526_6 use 400 fc and multi the cnn hidden number from 16,32 to 32,64
# 526_7 use 400fc and cnn hidden number is [10,20]
# 526_8 is same as 526_4 but 1024fc

log_dir = './log/531/5'
# 530_3 is same as 526_4
# 530_4 is same as 526_4 but fc has 200 instead of 400
# 530_5 use 400fc double cnn hidden
# 530_6 same as above but cnn1 kernal size [3,3]
# 530_7 abondon all cnn
# 530 8 use 2 hidden layer mlp

batch_size = 14

sign_dic = {
    './data/number/0': 0,
    './data/number/1': 1,
    './data/number/2': 2,
    './data/number/3': 3,
    './data/number/4': 4,
    './data/number/5': 5,
    './data/number/6': 6,
    './data/number/7': 7,
    './data/number/8': 8,
    './data/number/9': 9,
    './data/sign/dot': 13,
    './data/sign/minus': 11,
    './data/sign/multi': 12,
    './data/sign/plus': 10
}

new_dic = {
    './data/new_number/0': 0,
    './data/new_number/1': 1,
    './data/new_number/2': 2,
    './data/new_number/3': 3,
    './data/new_number/4': 4,
    './data/new_number/5': 5,
    './data/new_number/6': 6,
    './data/new_number/7': 7,
    './data/new_number/8': 8,
    './data/new_number/9': 9,
}


def next_batch(size, private_data=False):
    if private_data:
        num_dir = './data/new_number'
        dic = new_dic
    else:
        num_dir = './data/number'
        dic = sign_dic

    img = np.empty([size, 784])
    sign_lab = []
    cnt = 0

    for root, dir, files in os.walk(num_dir):
        if dir != []:
            continue
        files.pop(0)
        lis = np.random.choice(files, 1)
        for j in range(1):
            file_path = os.path.join(root, lis[j])
            img[cnt] = (np.resize(cv2.imread(file_path, 0), (784)))
            sign_lab.append(dic[root])
            cnt += 1
    for i in range(10):
        _, thr = cv2.threshold(np.resize(img[i] * 255, [28, 28]), 128, 255, cv2.THRESH_BINARY)
        img[i] = np.resize(thr, [1, 784])
    for root, dir, files in os.walk('./data/sign'):
        if dir != []:
            continue
        files.pop(0)
        lis = np.random.choice(files, 1)
        for j in range(1):
            file_path = os.path.join(root, lis[j])
            img[cnt] = (np.resize(cv2.imread(file_path, 0), [784]))
            sign_lab.append(sign_dic[root])
            cnt += 1
    onehot_lab = np.zeros((size, 14))
    for i in range(size):
        onehot_lab[i][sign_lab[i]] = 1
    return img, onehot_lab


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义placeholoder
x = tf.placeholder(tf.float32, [None, 784])
with tf.name_scope('input'):
    y_ = tf.placeholder(tf.float32, [None, 14])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('Input', x_image, 1)

# 卷积层
with tf.name_scope('CNN_1'):
    w_conv1 = weight_variable([5, 5, 1, 10])
    b_conv1 = bias_variable([10])
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # CNN输出结果的可视化
    # output_image = tf.split(h_pool1, 10, 3)
    # output_image = [tf.squeeze(i, 3) for i in output_image]
    # output_image = [tf.expand_dims(i, 3) for i in output_image]
    # output_image = tf.squeeze(output_image,1)
    # output_image_sum = [tf.summary.image('cnn1', output_image, 10)]

with tf.name_scope('CNN_2'):
    w_conv2 = weight_variable([3, 3, 10, 20])
    b_conv2 = bias_variable([20])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # CNN输出结果的可视化
    # output_image = tf.split(h_pool2, 20,3)
    # output_image = [tf.squeeze( i ,3) for i in output_image]
    # output_image = [tf.expand_dims(i,3) for i in output_image]
    # output_image = tf.squeeze(output_image, 1)
    # output_image_sum = [tf.summary.image('cnn2',output_image,20)]

# 全连接层
with tf.name_scope('Fully_Connected'):
    w_fc1 = weight_variable([7 * 7 * 20, 400])
    b_fc1 = bias_variable([400])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 20])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# dropout
with tf.name_scope('Drop_out'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    w_fc2 = weight_variable([400, 14])
    b_fc2 = bias_variable([14])

with tf.name_scope('Output'):
    y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2  # 此乃结果

# cost
with tf.name_scope('Cross_entropy'):
    # print(y_.shape,y_conv.shape)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_sum(cross_entropy)
tf.summary.scalar('cross_entropy', cross_entropy)

# 各种操作的存储
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
predict_op = tf.argmax(y_conv, 1)
tf.add_to_collection('predict_op', predict_op)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_dir, sess.graph)
tf.add_to_collection('x', x)
tf.add_to_collection('y_', y_)
tf.add_to_collection('keep_prob', keep_prob)

########################
# 训练过程
########################

tf.global_variables_initializer().run()
saver = tf.train.Saver()
# 读入文件
saver.restore(sess, './log/531/4/model.ckpt-16999')

for i in range(16000, 17000):
    batch = next_batch(batch_size)
    if i % 10 == 0:  # 每100次训练评测一次

        summary, acc = sess.run([merged, accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        writer.add_summary(summary, i)
        print('step %d, training accuracy %g, time %g' % (i, acc, time.time() - tmptime))
        tmptime = time.time()

    if i % 100 == 99:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        writer.add_run_metadata(run_metadata, 'step%03d' % i)
        saver.save(sess, log_dir + '/model.ckpt', i)
        print('save metadata in ', i)

    if i % 100 != 0:
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.75})

writer.close()

# 本类打开训练好的网络, 并向上级类提供操作网络的接口。
import tensorflow as tf

class Network(object):

    def __init__(self):
        self.sess = tf.InteractiveSession()

        # 打开训练好的网络
        new_saver = tf.train.import_meta_graph('./network/model.ckpt-16999.meta')
        new_saver.restore(self.sess, './network/model.ckpt-16999')

        # 引入网络中的op和placeholder
        self.predict = tf.get_collection('predict_op')[0]
        self.x = tf.get_collection('x')[0]
        self.y_ = tf.get_collection('y_')[0]
        self.k = tf.get_collection('keep_prob')[0]

    # 识别单个字符。
    def eval(self,img):
        return self.sess.run(self.predict,feed_dict={self.x:img,self.k:1.0})

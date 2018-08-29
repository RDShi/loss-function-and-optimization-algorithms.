import tflearn
import tensorflow as tf

def inference(incoming):
    with tf.variable_scope('conv1_1', reuse=tf.AUTO_REUSE): # 28*28*3->28*28*32
        net = tflearn.conv_2d(incoming, nb_filter=32, filter_size=3, bias=False, regularizer='L2', weight_decay=0.0001)
        net = tflearn.batch_normalization(net)
        net = tf.nn.relu(net)
    with tf.variable_scope('conv1_2', reuse=tf.AUTO_REUSE): # 28*28*32->28*28*32
        net = tflearn.conv_2d(net, nb_filter=32, filter_size=3, bias=False, regularizer='L2', weight_decay=0.0001)
        net = tflearn.batch_normalization(net)
        net = tf.nn.relu(net)
    print(net)
    net = tflearn.max_pool_2d(net, kernel_size=2, strides=2, padding='valid', name='pool1')
    print(net)
    
    with tf.variable_scope('conv2_1', reuse=tf.AUTO_REUSE): # 14*14*32->14*14*64
        net = tflearn.conv_2d(net, nb_filter=64, filter_size=3, bias=False, regularizer='L2', weight_decay=0.0001)
        net = tflearn.batch_normalization(net)
        net = tf.nn.relu(net)
    with tf.variable_scope('conv2_2', reuse=tf.AUTO_REUSE): # 14*14*64->14*14*64
        net = tflearn.conv_2d(net, nb_filter=64, filter_size=3, bias=False, regularizer='L2', weight_decay=0.0001)
        net = tflearn.batch_normalization(net)
        net = tf.nn.relu(net)
    net = tflearn.max_pool_2d(net, kernel_size=2, strides=2, padding='valid', name='pool2')
    print(net)
    
    with tf.variable_scope('conv3_1', reuse=tf.AUTO_REUSE): # 7*7*64->7*7*128
        net = tflearn.conv_2d(net, nb_filter=128, filter_size=3, bias=False, regularizer='L2', weight_decay=0.0001)
        net = tflearn.batch_normalization(net)
        net = tf.nn.relu(net)
    with tf.variable_scope('conv3_2', reuse=tf.AUTO_REUSE): # 7*7*128->7*7*128
        net = tflearn.conv_2d(net, nb_filter=128, filter_size=3, bias=False, regularizer='L2', weight_decay=0.0001)
        net = tflearn.batch_normalization(net)
        net = tf.nn.relu(net)
    net = tflearn.global_avg_pool(net)
    feature = tflearn.fully_connected(net, n_units=2, regularizer='L2', weight_decay=0.0001)
    print(net)
    
    net = tflearn.prelu(feature)
    net = tflearn.fully_connected(net, n_units=10, regularizer='L2', weight_decay=0.0001)
    
    return net, feature


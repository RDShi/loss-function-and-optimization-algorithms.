import tensorflow as tf
import numpy as np

def get_center_loss(features, labels, alpha=0.5, num_classes=10):
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels) # each center for each feature
    loss = tf.nn.l2_loss(features - centers_batch)

    diff = centers_batch - features
    _, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx) # appear times of each center for each feature
    appear_times = tf.reshape(appear_times, [-1, 1])
    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    centers_update_op = tf.scatter_sub(centers, labels, diff) # labels is index: centers[labels]-diff
    
    return loss, centers_update_op

def similar_matrix(embeddings, centers):
    
    dot_product = tf.matmul(embeddings, tf.transpose(centers))
    centers_norm = tf.sqrt(tf.reduce_sum(tf.square(centers), axis=1)) + 1e-16
    cosin = dot_product/centers_norm

    return cosin

    # W = tf.Variable(np.random.randn(), name="weight", dtype=tf.float32)
    # b = tf.Variable(np.random.randn(), name="b", dtype=tf.float32)
    # similarity = W*cosin+b
    
    # return similarity


def get_te2e_loss(embeddings, labels, alpha=0.5, pho=1.0, num_classes=10):

    labels = tf.reshape(labels, [-1])

    center_list = []
    for i in range(num_classes):
        mask = tf.transpose(tf.concat([[tf.to_float(tf.equal(labels, i))],[tf.to_float(tf.equal(labels, i))]], axis=0))
        center_i = tf.reduce_sum(mask*embeddings, axis=0)/(tf.reduce_sum(mask)/2+1e-16)
        center_list.append([center_i])
    centers = tf.concat(center_list,axis=0)
    similarity = similar_matrix(embeddings, centers)
    loss_type1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=similarity))
    
    onehot = tf.one_hot(labels, num_classes)
    loss_type2 = 1 - tf.reduce_mean(tf.nn.sigmoid(tf.reduce_sum(onehot*similarity))) \
                 + tf.reduce_mean(tf.nn.sigmoid(tf.reduce_max((1-onehot)*similarity)))
    
    loss = pho*loss_type1 + (1.0-pho)*loss_type2
 
    return loss

def train_op(labels, logits, features, embeddings, ratio_s=1, ratio_c=0, ratio_t=0, ratio_g=0):

    with tf.name_scope('loss'):
        with tf.name_scope('center_loss'):
            center_loss, centers_update_op = get_center_loss(features, labels)
        with tf.name_scope('ge2e_loss'):
            ge2e_loss = get_te2e_loss(embeddings, labels, pho=0.0)
        with tf.name_scope('triplet_loss'):
            triplet_loss = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, features, margin=1.0)
        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        with tf.name_scope('l2_loss'):
            l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')
        with tf.name_scope('total_loss'):
            total_loss = ratio_s*softmax_loss + l2_loss + ratio_c * center_loss + ratio_t * triplet_loss + ratio_g * ge2e_loss
    
    optimizer = tf.train.AdamOptimizer(0.001)
    global_step = tf.Variable(0, trainable=False, name='global_step')
    with tf.control_dependencies([centers_update_op]):
        trainer = optimizer.minimize(total_loss, global_step=global_step)

    return trainer, total_loss


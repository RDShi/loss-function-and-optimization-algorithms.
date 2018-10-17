import sys
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import tensorflow as tf
import tflearn
import numpy as np
import matplotlib.pyplot as plt

from data_prepare import mnist
from model import inference
mnist = mnist()

opt_type = sys.argv[1]
log_dir = 'logdir/'+opt_type

tf.reset_default_graph()
x = tf.placeholder(tf.float32,[None, 28,28,1])
y = tf.placeholder(tf.int64,[None])
lr_placeholder = tf.placeholder(tf.float32)

logits, _ = inference(x)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
    tf.summary.scalar('train', loss)

print(opt_type)
if opt_type == 'Adadelta':
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-06, name='Adadelta')
elif opt_type == 'Adam':
    optimizer = tf.train.AdamOptimizer(0.001)
elif opt_type == 'Adagrad':
    # optimizer = tf.train.AdagradOptimizer(learning_rate=lr_placeholder)
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
elif opt_type == 'Ftrl':
    optimizer = tf.train.FtrlOptimizer(learning_rate=lr_placeholder)
elif opt_type == 'SGD':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
elif opt_type == 'Momentum':
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=False)
elif opt_type == 'Nesterov':
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, use_nesterov=True)
elif opt_type == 'RMSprop':
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
else:
    raise ValueError

global_step = tf.Variable(0, trainable=False, name='global_step')
trainer = optimizer.minimize(loss, global_step=global_step)

with tf.name_scope('acc'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), y), tf.float32))
    tf.summary.scalar('train', accuracy)

merged = tf.summary.merge_all()

mean_data = np.mean(mnist.train.images, axis=0)

config = tf.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
    
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    tflearn.is_training(True, session=sess)

    while step <= 20000:
        with open('data/learningrate.txt') as fid:
            lr = float(fid.readline()) 
        batch_images, batch_labels = mnist.train.next_batch(128)
        _, train_acc, train_loss, result = sess.run([trainer, accuracy, loss, merged], feed_dict={x: batch_images - mean_data, y: batch_labels, lr_placeholder: lr})
        step += 1
        
        train_writer.add_summary(result, step)
    
        if step % 100 == 0:
            tflearn.is_training(False, session=sess)
            vali_image = mnist.validation.images - mean_data
            vali_acc, vali_loss = sess.run([accuracy, loss], feed_dict={x: vali_image, y: mnist.validation.labels, lr_placeholder: lr})
            print(("step: {}, train_acc:{:.4f}, train_loss:{:.4f}, vali_acc:{:.4f}, test_loss:{:.4f}, learning rate:{}".
                   format(step, train_acc, train_loss, vali_acc, vali_loss, lr)))
            summary = tf.Summary()
            summary.value.add(tag='acc/val', simple_value=vali_acc)
            summary.value.add(tag='loss/val', simple_value=vali_loss)
            train_writer.add_summary(summary, step)
            tflearn.is_training(True, session=sess)
  


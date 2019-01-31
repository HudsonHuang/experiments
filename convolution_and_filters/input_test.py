# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 17:57:37 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

os.environ['CUDA_VISIBLE_DEVICES'] = ''

a = np.random.random((1,1024,1))

inpu = tf.constant(a,dtype=tf.float32)
k_np  = np.random.random((15,1,1)).astype(np.float32)
k = tf.Variable(k_np)

#kernel_initializer=k,bias_initializer=b,

y = tf.nn.conv1d(inpu,k,padding='SAME',stride=1)

loss = tf.losses.mean_squared_error(labels=a,predictions=y)

train_op  = tf.train.AdamOptimizer(1e-3).minimize(loss)
ims = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        _,k_img,l,y_hat = sess.run([train_op,k,loss,y])
        k_img = k_img.flatten()

        if i%1000 == 0:
            plt.ylim(0.0,1.0)
            plt.plot(k_img)
            plt.show()
            print("loss = {}".format(l))
#            im = plt.imshow(plt.plot(k_img), animated=True)
#            ims.append([im])
#
#
#ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
#                                repeat_delay=1000)
#
#ani.save('dynamic_images.mp4')
#plt.show()
final = np.convolve(a.flatten(),k_img,mode='same')
#filter = tf.Variable(tf.truncated_normal([8, 8, 3,1]))
#images = tf.placeholder(tf.float32, shape=[None, 28, 28,1])
#
#conv = tf.nn.conv2d(images, filter, strides=[1, 1, 1, 1], padding="SAME")
#
## More ops...
#loss = ...
#optimizer = tf.GradientDescentOptimizer(0.01)
#train_op = optimizer.minimize(loss)
#
#filter_summary = tf.image_summary(filter)
#
#sess = tf.Session()
#summary_writer = tf.train.SummaryWriter('/tmp/logs', sess.graph_def)
#for i in range(10000):
#  sess.run(train_op)
#  if i % 10 == 0:
#    # Log a summary every 10 steps.
#    summary_writer.add_summary(filter_summary, i)
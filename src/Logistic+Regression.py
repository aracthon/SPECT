
# coding: utf-8

# In[114]:

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[115]:

src_path = os.path.abspath(os.path.dirname(sys.argv[0]))
data_path = '../data/'

training_data = np.loadtxt(data_path + 'SPECT.train.txt', delimiter=',');
test_data = np.loadtxt(data_path + 'SPECT.test.txt', delimiter=',');

# Format the data
# trainX: 80x22, testX:187x22

trY = training_data[:,0].reshape(80, 1)
trX = training_data[:,1:23]

teY = test_data[:,0].reshape(187, 1)
teX = test_data[:,1:23]

# print teX.shape
# print teY.shape
# print trX.shape
# print trY.shape


# In[116]:

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w):
    return tf.matmul(X, w)


# In[139]:

X = tf.placeholder("float", [None, 22])
Y = tf.placeholder("float", [None, 1])

w = init_weights([22, 1])

pred = model(X, w)

learning_rate = 0.1
training_epochs = 100

init = tf.global_variables_initializer()
cost = -tf.reduce_sum(Y * tf.log(pred))
# tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(teY, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# In[140]:

cost_history = np.empty(shape=[1],dtype=float)

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in xrange(training_epochs):
        sess.run(optimizer, feed_dict={X: trX, Y: trY})
        cost_history = np.append(cost_history, sess.run(cost,
                            feed_dict={X: trX, Y: trY}))
      # if epoch % 100 == 0:
            #     print sess.run(cost, feed_dict={X: trX[start:end], Y: trY[start:end]})
    print(sess.run(accuracy,
               feed_dict={
                   X: teX,
               }))

    fig = plt.figure(figsize=(10,8))
    plt.plot(cost_history)
    plt.axis([0,training_epochs,0,np.max(cost_history)])
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.show()

# In[ ]:




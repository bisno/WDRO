# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp
from cvxopt import solvers
import matplotlib.pyplot as plt

tf.reset_default_graph()
np.random.seed(10)
rand_x = np.random.randn(1500)/50
np.random.seed(8)
rand_y = np.random.randn(1500)/50
# rand_y = np.random.randn(1500)/50
solvers.options['show_progress'] = False

other = [(1.23, 3.01),(0.98, 3.32),(1.77, 3.92),(1.48, 4.52),(0.63, 2.89), (1.92, 5.0), (1.1, 2.8),(0.71, 3.17),
         (1.64, 4.54),(1.26, 3.96),(1.22, 2.84), (0.77, 2.59),(1.89, 5.1),(1.13,3.17), (1.31, 2.91)]

u2 = np.zeros((1515,1))
v2 = np.zeros((1515,1))
for i in range(500):
    u2[i],v2[i] = 0.16+rand_x[i], 1.22+rand_y[i]
for i in range(500):
    u2[i+500],v2[i+500] = 0.43+rand_x[i+500],1.45+rand_y[i+500]
for i in range(500):
    u2[i+1000],v2[i+1000] = 0.04+rand_x[i+1000],1.59+rand_y[i+1000]
for i in range(15):
    u2[i+1500],v2[i+1500] = other[i][0],other[i][1]

# Separate dataset into two subgroups.
X1 = tf.constant(u2[:1500])
y1 = tf.constant(v2[:1500])
X2 = tf.constant(u2[1500:])
y2 = tf.constant(v2[1500:])

w = tf.get_variable("w", shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer(),dtype='float64')
b = tf.Variable(tf.constant(0.1, shape=[1],dtype='float64'))

z1 = tf.reduce_mean(tf.square(tf.matmul(X1,w)+b - y1))
z2 = tf.reduce_mean(tf.square(tf.matmul(X2,w)+b - y2))

# Define the max_mean of each subgroup's loss
# according to equation (1).
z = tf.maximum(z1,z2)

z1_grad = tf.gradients(ys=z1,xs=w)
z2_grad = tf.gradients(ys=z2,xs=w)

z1_grad_b = tf.gradients(ys=z1,xs=b)
z2_grad_b = tf.gradients(ys=z2,xs=b)

# WDRO = []
# MSE = []

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('start...')
for i in range(300):
         
    # Compute the gradient of 'w'.
    GG = np.zeros([2,1])
    hh = np.zeros(2)
    g1 = sess.run(z1_grad)
    g2 = sess.run(z2_grad)    
    GG[0,:] = g1[0].reshape(-1)
    GG[1,:] = g2[0].reshape(-1)
    hh[0],hh[1] = sess.run([z1,z2])
        
    P = matrix(GG)*matrix(GG).T        
    q = -matrix(hh)
    G = matrix(-np.eye(2))
    h = matrix(np.zeros(2))
    A = matrix(np.ones([1,2]))
    b_matrix = matrix(np.ones([1]))
    
    # Solve quadratic programming of the equation (4)-(5).
    res = qp(P,q,G=G,h=h,A=A,b=b_matrix)
    
    # Get descent direction.
    d = -np.array(GG).T.dot(np.array(res['x']))[:,0].reshape(-1,1)

    now = sess.run(z)
    ww = sess.run(w)
    t = 10
         
    # This part is optional: 
    # The implementation of line-search.
    for j in range(100):
        if sess.run(z,feed_dict={w:ww+t*d}) < now:
            break
        t = 0.8*t
    
    sess.run(w.assign(ww+t*d))

    # Compute the gradient of 'b'.
    # This part is similar to get the descent direction of 'w'.
    GG = np.zeros([2,1])
    hh = np.zeros(2)
    g1 = sess.run(z1_grad_b)
    g2 = sess.run(z2_grad_b)    
    GG[0,:] = g1[0].reshape(-1)
    GG[1,:] = g2[0].reshape(-1)
    hh[0],hh[1] = sess.run([z1,z2])
        
    P = matrix(GG)*matrix(GG).T        
    q = -matrix(hh)
    G = matrix(-np.eye(2))
    h = matrix(np.zeros(2))
    A = matrix(np.ones([1,2]))
    b_matrix = matrix(np.ones([1]))
    
    
    res = qp(P,q,G=G,h=h,A=A,b=b_matrix)
    db = -np.array(GG).T.dot(np.array(res['x']))[:,0]
    bb = sess.run(b)
    sess.run(b.assign(bb+0.1*db))
    
    
    cost11 = sess.run(z)
    print("epoch =",i+1,", WDRO =",cost11,", stepsize =",t)

wWDRO,bWDRO = sess.run(w)[0], sess.run(b)

tf.reset_default_graph()

x2 = np.zeros((1515,1))
y2 = np.zeros((1515,1))
for i in range(500):
    x2[i],y2[i] = 0.16+rand_x[i], 1.22+rand_y[i]
for i in range(500):
    x2[i+500],y2[i+500] = 0.43+rand_x[i+500],1.45+rand_y[i+500]
for i in range(500):
    x2[i+1000],y2[i+1000] = 0.04+rand_x[i+1000],1.59+rand_y[i+1000]
for i in range(15):
    x2[i+1500],y2[i+1500] = other[i][0],other[i][1]
    
x = tf.placeholder(tf.float32, [None, 1], name='x') 
y = tf.placeholder(tf.float32, [None, 1], name='y')

W = tf.get_variable("w", shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.constant(0.1, shape=[1]))
init = tf.global_variables_initializer()
pred = tf.matmul(x, W) +b
cost = tf.reduce_mean(tf.square(pred-y))
opt = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(init)
    for i in range(300):
        feed_dicts = {x:x2,
                     y:y2}
        feed_dicts_n = {x:x2[:1500],
                     y:y2[:1500]}
        feed_dicts_p = {x:x2[1500:],
                     y:y2[1500:]}
        cosss,_ = sess.run((cost,opt), feed_dict=feed_dicts)
        cost_n = sess.run(cost,feed_dict=feed_dicts_n)
        cost_p = sess.run(cost,feed_dict=feed_dicts_p)

        cost_max = np.max([cost_n,cost_p])
        
        print("epoch =",i+1,", loss =",cosss, "WDRO = ",cost_max)

    wout,bout = sess.run(W),sess.run(b)

plt.figure(figsize=(10,8))
group1_x = u2[:1500]
group1_y = v2[:1500]
group2_x = u2[1500:]
group2_y = v2[1500:]
# plt.scatter(kk1,pp1,color = 'r')
plt.scatter(group1_x,group1_y,color = '#4682B4')
plt.scatter(group2_x,group2_y,s=10)

plt.title('Linear Regression Example',fontdict={'weight':'normal','size': 20})

plt.plot([-0.4,3],[-0.4*wout[0] + bout, 3*wout[0] + bout],linewidth=1.0,color='coral',linestyle='-')
plt.plot([-0.4,3],[-0.4*wWDRO+bWDRO, 3*wWDRO+bWDRO],linewidth=1.0,color='#4682B4',linestyle='-')
plt.legend(['AvgLoss','MaxLoss'],fontsize=20)
plt.tick_params(labelsize=16)
plt.savefig('./outcome/regression_WDRO_compare.png', dpi=300)

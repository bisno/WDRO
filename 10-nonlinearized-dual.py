# -*- coding: utf-8 -*-


import tensorflow as tf
import tensorflow.contrib.slim as slim

import glob
import numpy as np
import pandas as pd
from PIL import Image
import random
import time
from data_process import *
from model import *
import sys

from cvxopt import matrix
from cvxopt.solvers import qp
from cvxopt import solvers

import argparse

parser = argparse.ArgumentParser(description='sIR and subgroup batch_size')
parser.add_argument('--num_of_group', type=int, default=4)
parser.add_argument('--num_minor', type=int, default=2)
parser.add_argument('--train_subgroup_batch', type=int, default=100)

args = parser.parse_args()
num_of_group = args.num_of_group
num_correct = args.num_minor
train_subgroup_batch = args.train_subgroup_batch

solvers.options['show_progress'] = False
date = time.strftime("%Y%m%d%H%M%S", time.localtime())


training_epochs = 2000
test_batch_num = 100
display_step = 1

m = 2 * size

channel = 3

# tf Graph Input
x = tf.placeholder(tf.float32, [None, size*3], name='x')
y = tf.placeholder(tf.float32, [None, 2], name='y')

filter_size1 = 3
num_filters1 = 16

filter_size2 = 3
num_filters2 = 36


final_out_n = 3000

layer_fc1, w1 = new_fc_layer(name="w1",
                             input=x,
                             num_inputs=9000,
                             num_outputs=3000,
                             use_relu=True)

layer_fc2, w2 = new_fc_layer(name="w2",
                             input=layer_fc1,
                             num_inputs=3000,
                             num_outputs=final_out_n,
                             use_relu=True)


W = tf.get_variable("w", shape=[final_out_n, 2],
                    initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.constant(0.01, shape=[2]))


pred = tf.nn.softmax(tf.matmul(layer_fc2, W) + b)  # Softmax
tf.add_to_collection('pred', pred)


def cost_n_to_list(n, pred):
    list_ = []
    for _ in range(n):
        list_.append(tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)))
    return list_


def max_tensor(cost_combine):
    for i in range(len(cost_combine)-1):
        if i == 0:
            cost = cost_combine[i]
        else:
            cost = tf.maximum(cost, cost_combine[i+1])
    return cost


def grad_combine(cost_combine, var):
    gradss = []
    for i in range(len(cost_combine)):
        gradss.append(tf.gradients(xs=[var], ys=cost_combine[i]))
    return gradss


cost_combine = cost_n_to_list(num_of_group, pred)
cost_final = max_tensor(cost_combine)

grad_combine_ = grad_combine(cost_combine, var=W)
grad_combine_b = grad_combine(cost_combine, var=b)


def get_Gh(grad_list, cost_list, m):
    N = len(cost_list)
    G = np.zeros([N, m])
    b = []

    for i in range(N):

        g = grad_list[i][0].flatten()
        G[i][:] = g

        b.append(float(cost_list[i]))  # add cost

    b = np.array(b)

    GG = matrix(G)
    hh = matrix(b)

    return GG, hh


def cal_grad(grad_list, cost_list, m, size_in, size_out):

    N = len(cost_list)

    GG, hh = get_Gh(grad_list, cost_list, m)
    P = matrix(GG)*matrix(GG).T
    q = -matrix(hh)
    G = matrix(-np.eye(N))
    h = matrix(np.zeros(N))
    A = matrix(np.ones([1, N]))
    b = matrix(np.ones([1]))
    res = qp(P, q, G=G, h=h, A=A, b=b)

    d = -np.array(GG).T.dot(np.array(res['x'])
                            )[:, 0].reshape(size_in, size_out)

    return d


cost21 = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))

grad_w1 = tf.gradients(xs=[w1], ys=cost21)
grad_w2 = tf.gradients(xs=[w2], ys=cost21)


def grad_cost_list(sess, cost_combine, grad_combine_, grad_combine_b, num_correct=num_correct):
    grad_list = []
    grad_list_b = []

    cost_list = []

    n = len(cost_combine)
    train_x = []
    train_y = []
#     print(n)
    for i in range(n):
        if i < n - num_correct:
            batch_xs, batch_ys, _ = create_trainbatch(
                train_subgroup_batch, channel)
            if channel != 0:
                batch_xs = batch_xs.reshape(-1, 3*size)
            c, g_W, g_b = sess.run([cost_combine[i], grad_combine_[
                                   i], grad_combine_b[i]], feed_dict={x: batch_xs, y: batch_ys})

            grad_list.append(g_W)
            grad_list_b.append(g_b)

            cost_list.append(c)
            train_x.append(batch_xs)
            train_y.append(batch_ys)

        else:

            batch_xs, batch_ys, _ = create_trainbatch_all_correct(
                train_subgroup_batch, channel)
            if channel != 0:
                batch_xs = batch_xs.reshape(-1, 3*size)
            c, g_W, g_b = sess.run([cost_combine[i], grad_combine_[
                                   i], grad_combine_b[i]], feed_dict={x: batch_xs, y: batch_ys})

            grad_list.append(g_W)
            grad_list_b.append(g_b)

            cost_list.append(c)
            train_x.append(batch_xs)
            train_y.append(batch_ys)

    return np.array(train_x).reshape(n*train_subgroup_batch, size*3), np.array(train_y).reshape(n*train_subgroup_batch, 2), grad_list, grad_list_b, cost_list


correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy_count = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
cost = tf.reduce_mean(-tf.reduce_sum(y *
                                     tf.log(tf.clip_by_value(pred, 1e-8, 1.0)), reduction_indices=1))


def print_accuracy(accuracy, accuracy_count, cost):
    num = 200
    batch_xx, batch_yy, _, d = create_testset(num, channel)
    if channel != 0:
        batch_xx = batch_xx.reshape(-1, size*3)

    batch_xx_correct = batch_xx[:num]
    batch_yy_correct = batch_yy[:num]
    batch_xx_false = batch_xx[num:]
    batch_yy_false = batch_yy[num:]

    feed_dict_test_T = {x: batch_xx_correct,
                        y: batch_yy_correct}

    feed_dict_test_F = {x: batch_xx_false,
                        y: batch_yy_false}

    feed_dict_test = {x: batch_xx,
                      y: batch_yy}

    TP = accuracy_count.eval(feed_dict=feed_dict_test_T)
    FN = len(batch_xx_correct) - TP
    TN = accuracy_count.eval(feed_dict=feed_dict_test_F)
    FP = len(batch_xx_false) - TN

    recall = TP/(TP + FN + 1e-8)
    precision = TP/(TP + FP + 1e-8)
    F1 = 2 * ((precision*recall)/(precision + recall + 1e-8))
    print(TP, FN, TN, FP)
    print(recall, precision, F1, '\n')

    cost_f = cost.eval(feed_dict=feed_dict_test)

    batch_x, batch_y, _ = create_trainbatch_(200, channel)
    if channel != 0:
        batch_x = batch_x.reshape(-1, size*3)
    feed_dict = {x: batch_x,
                 y: batch_y}
    accuracy_train = accuracy.eval(feed_dict=feed_dict)

    return cost_f, F1, accuracy_train, d


acc_list = [0]
acc_train_list = []
costs_list = []


optimizer = tf.train.GradientDescentOptimizer(0.0001)

grads_and_vars_all = optimizer.compute_gradients(cost21)[:-2]
print(grads_and_vars_all, '\n')


training_op_all = optimizer.apply_gradients(grads_and_vars_all)


def exchange(grads_and_vars, grad_, t):
    for i, (g, v) in enumerate([grads_and_vars[-1]]):

        if g is not None:
            grads_and_vars[-1] = (-grad_*t*100, v)  # exchange gradients


def scale_t(max_grad):
    n = 0
    num = 0.01
    if max_grad < num:
        while max_grad < num:
            max_grad = max_grad * 10
            n += 1
    return n


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    tf.global_variables_initializer().run()
    t = 1.0

    m_saver = tf.train.Saver()

    print('label_cls :  ', label_cls)
    for epoch in range(training_epochs):

        for i in range(1):

            batch_xs21, batch_ys21, grad_list, grad_list_b, cost_list = grad_cost_list(
                sess, cost_combine, grad_combine_, grad_combine_b)

            grad_ = cal_grad(grad_list, cost_list, m=final_out_n *
                             2, size_in=final_out_n, size_out=2)
            grad_b = cal_grad(grad_list_b, cost_list,
                              m=2, size_in=1, size_out=2)
            grad_ = grad_.astype(np.float32)
            grad_b = grad_b.astype(np.float32).reshape(2)

            print(np.max(grad_), np.max(grad_b))
            print(np.min(abs(grad_)), np.min(abs(grad_b)))
            print('iter_{}  curr(F1_list) : '.format(epoch), acc_list[-1])
            print('iter_{}  mean(F1_list) : '.format(
                epoch), np.mean(acc_list[-10:-1]))
            print('iter_{}  max (F1_list) : '.format(epoch), max(acc_list))

            training_op = optimizer.apply_gradients(
                [((-grad_*t*10000).astype(np.float32), W)])
            training_op_b = optimizer.apply_gradients(
                [((-grad_b*t*100).astype(np.float32), b)])

            sess.run(training_op, feed_dict={x: batch_xs21, y: batch_ys21})
            sess.run(training_op_b, feed_dict={x: batch_xs21, y: batch_ys21})
            sess.run(training_op_all, feed_dict={x: batch_xs21, y: batch_ys21})

        if (epoch+1) % display_step == 0:

            costs, acc, acc_train, _ = print_accuracy(
                accuracy, accuracy_count, cost)
            acc_list.append(acc)
            costs_list.append(costs)
            acc_train_list.append(acc_train)

            # np.savez('./result/{}_F1_{}_{}.npz'.format(label_cls,num_of_group,date),test = acc_list)

            print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.5f}".format(costs), "stepsize = {:.4f}".format(
                t), "acc_test_f1 = {:.4f}".format(acc), "acc_train =", acc_train)
            # if (epoch) % 5 == 0:
            #     m_saver.save(sess, "models/MER", global_step=epoch)

    print("Optimization Finished!")

#this file is part of litwin-kumar_et_al_dimension_2017
#Copyright (C) 2017 Ashok Litwin-Kumar
#see README for more information

import time
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#computes the classification accuracy after learning of both input to hidden layer and hidden layer to output connections
def calcacc(K,N,M,P,Nex,batchsize,maxsteps,dobias=True):
    t = time.time()

    eps = 0.5

    tf.reset_default_graph()

    Jmask = np.zeros([N,M])
    for mi in range(M):
        inds = np.random.choice(range(N),K,replace=False)
        Jmask[inds,mi] = 1

    #placeholders
    s = tf.placeholder(tf.float32,shape=[None,N])
    y = tf.placeholder(tf.float32,shape=[None,2])

    #variables
    J = tf.Variable(tf.random_normal(shape=[N,M],mean=0.,stddev=1./np.sqrt(K)))
    w = tf.Variable(tf.random_normal(shape=[M,2],mean=0.,stddev=1./np.sqrt(M)))
    if dobias:
        bias = tf.Variable(tf.zeros([1,M]))
        biasout = tf.Variable(tf.zeros([1,2]))

    if dobias:
        h = tf.nn.relu(bias + tf.matmul(s,Jmask*J))
        out = biasout + tf.matmul(h,w)
    else:
        h = tf.nn.relu(tf.matmul(s,Jmask*J))
        out = tf.matmul(h,w)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out,labels=y))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    pats = np.random.randn(P,N)
    labels = np.random.randint(0,2,P)
    labelvec = np.zeros([P,2])
    labelvec[range(P),labels] = 1

    train_x = np.repeat(pats,Nex,axis=0) + eps*np.random.randn(P*Nex,N)
    train_y = np.repeat(labelvec,Nex,axis=0)


    sess = tf.InteractiveSession()
    init = tf.initialize_all_variables()
    sess.run(init)

    #acc = np.zeros(maxsteps)
    for ti in range(maxsteps):
        print('\r',100*ti/maxsteps,'%',end='')
        batchinds = np.random.choice(range(P*Nex),batchsize,replace=False)
        x_batch = train_x[batchinds,:]
        y_batch = train_y[batchinds,:]
        sess.run(train_step,feed_dict={s: x_batch, y: y_batch})

    test_x = pats + eps*np.random.randn(P,N)
    acc = sess.run(accuracy,feed_dict={s: test_x, y: labelvec})
    sess.close()


    print("\relapsed:",time.time() - t)
    return acc



#computes the classification accuracy after learning of hidden layer to output connections, with fixed input to hidden layer connections
def calcacc_fixed(K,N,M,P,Nex,batchsize,maxsteps,dobias=True):
    t = time.time()

    eps = 0.5

    tf.reset_default_graph()

    J = np.zeros([N,M],dtype=np.float32)
    for mi in range(M):
        inds = np.random.choice(range(N),K,replace=False)
        J[inds,mi] = np.random.randn(K)
        J[:,mi] = J[:,mi] / np.sqrt(np.sum(J[:,mi]*J[:,mi]))

    #placeholders
    s = tf.placeholder(tf.float32,shape=[None,N])
    y = tf.placeholder(tf.float32,shape=[None,2])

    #variables
    w = tf.Variable(tf.random_normal(shape=[M,2],mean=0.,stddev=1./np.sqrt(M)))
    if dobias:
        bias = tf.Variable(tf.zeros([1,M]))
        biasout = tf.Variable(tf.zeros([1,2]))

    if dobias:
        h = tf.nn.relu(bias + tf.matmul(s,J))
        out = biasout + tf.matmul(h,w)
    else:
        h = tf.nn.relu(tf.matmul(s,J))
        out = tf.matmul(h,w)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=out,labels=y))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    pats = np.random.randn(P,N)
    labels = np.random.randint(0,2,P)
    labelvec = np.zeros([P,2])
    labelvec[range(P),labels] = 1

    train_x = np.repeat(pats,Nex,axis=0) + eps*np.random.randn(P*Nex,N)
    train_y = np.repeat(labelvec,Nex,axis=0)


    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    for ti in range(maxsteps):
        print('\r',100*ti/maxsteps,'%',end='')
        batchinds = np.random.choice(range(P*Nex),batchsize,replace=False)
        x_batch = train_x[batchinds,:]
        y_batch = train_y[batchinds,:]
        sess.run(train_step,feed_dict={s: x_batch, y: y_batch})

    test_x = pats + eps*np.random.randn(P,N)
    acc = sess.run(accuracy,feed_dict={s: test_x, y: labelvec})
    sess.close()


    print("\relapsed: ",time.time() - t)
    return acc

N = 500
M = 10000
Nex = 1
P = 500
batchsize = round(P/5)
maxsteps = 1000
Ntrials = 20
dobias = False
learninput = False

Ka = np.unique(np.round(np.logspace(0,np.log10(N),25)).astype(np.int32))
Nk = len(Ka)

err = np.zeros([Nk,Ntrials])

for ti in range(Ntrials):
    for ki in range(Nk):
        print("K =",Ka[ki],",trial ",ti+1)

        if learninput:
            err[ki,ti] = 1-calcacc(Ka[ki],N,M,P,Nex,batchsize,maxsteps,dobias)
        else:
            err[ki,ti] = 1-calcacc_fixed(Ka[ki],N,M,P,Nex,batchsize,maxsteps,dobias)

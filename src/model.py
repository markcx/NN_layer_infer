"""
This is time series convolution test
"""

import pandas as pd
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.preprocessing as skPre
import tensorflow as tf

import util_helper as util_h

share_name="AAPL"  #"IBM"
ts_raw_data = pd.read_csv("../simple_data/%s.csv"%share_name )

ts_raw_data["stock_val"]=ts_raw_data[["Open", "Close"]].mean(axis=1)

def visTS_Plot(ts_raw_data, label="IBM"):
    fig, ax = plt.subplots(figsize=(9,6.5))
    ts = pd.Series( ts_raw_data["stock_val"].values, index=ts_raw_data["Date"].values)
    ts.plot(label=label)
    ax.legend()
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=30, fontsize=11)
    plt.show()

# visTS_Plot(ts_raw_data, share_name)

# #####################################
def next_batch(data_input, i, batch_size):
    #
    # assert(((i-2) * batch_size) > len(data_input), "Out of Range")
    #
    x_ = data_input[(i*batch_size):((i+1)*batch_size)]
    y_ = data_input[((i+1)*batch_size):((i+2)*batch_size)]

    return x_, y_


seq_length=5
x = tf.placeholder(shape=[None, 1, 5, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1, 5, 1], dtype=tf.float32)

y_hat = util_h.conv_bn_lrelu(x, kernel_shape=[1,1,1,1], bias_shape=[1,1,1,1] )

loss = tf.losses.mean_squared_error(y_hat, y_target)
# optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_opt = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    episodes = ts_raw_data.shape[0] // 5
    input_xs = ts_raw_data["stock_val"].values
    print(episodes)
    losses = []
    for e in range(2000):
        i_ = e % (episodes - 2)
        xs, ys = next_batch(input_xs, i_, batch_size=5)
        # print(e, xs, ys)
        xs_norm = (xs - xs.min(axis=0)) / (xs.max(axis=0) - xs.min(axis=0))
        ys_norm = (ys - ys.min(axis=0)) / (ys.max(axis=0) - ys.min(axis=0))
        # X_scaled = xs_norm * (max(xs) - min(xs)) + min(xs)
        # print(e, x_norm, X_scaled)
        xs = xs_norm[np.newaxis, np.newaxis, :, np.newaxis]
        ys = ys_norm[np.newaxis, np.newaxis, :, np.newaxis]
        _, _loss = sess.run([train_opt, loss], feed_dict={x: xs, y_target: ys})
        print("{:^7}{:^7.2f}".format(e , _loss))
        if e % 10 == 0 :
            losses.append(_loss)


    plt.plot(np.arange(len(losses)), losses)
    plt.show()
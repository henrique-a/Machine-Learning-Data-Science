import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data

n_steps = 28
n_inputs = 28
n_neurons = 128
n_outputs = 10 # number of classes 

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32)

cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

pred = fully_connected(states, n_outputs, activation_fn=None) # the states are the inputs
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
X_test = mnist.test.images.reshape(-1, n_steps, n_inputs)
y_test = mnist.test.labels

n_epochs = 10
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for i in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(optimizer, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})

        print('Epoch', epoch+1, 'Train accuracy:', acc_train, 'Test accuracy:', acc_test)


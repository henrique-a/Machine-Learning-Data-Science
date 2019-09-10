import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_epochs = 3 # Number of cycles (feed foward + backpropagation)
n_classes = 10
batch_size = 128  
chunk_size = 28
n_chunks = 28
rnn_size = 128


X = tf.placeholder(tf.float32, [None, n_chunks, chunk_size]) # Defines type and shape of tensor (28px * 28px)
y = tf.placeholder(tf.float32) # Defines type and shape of tensor

def recurrent_neural_network_model(X):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size,n_classes])),
    'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    X = tf.transpose(X, [1,0,2])
    X = tf.reshape(X, [-1, chunk_size])
    X = tf.split(0, n_chunks, X)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
    outputs, states = rnn.rnn(lstm_cell, X, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])
    
    return output

def train_neural_network(X):
    pred = recurrent_neural_network_model(X)
    # Compute the mean of probability error measures in discrete classification
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdadeltaOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                # Get the next batch of entries
                epoch_X, epoch_y = mnist.train.next_batch(batch_size)
                epoch_X = epoch_X.reshape((batch_size, n_chunks,chunk_size))
                # Train the nn with the batch of entries
                _, c = sess.run([optimizer, cost], feed_dict={X: epoch_X, y: epoch_y})
                epoch_loss += c
            
            print('Epoch', epoch + 1, 'completed out of', n_epochs, 'loss', epoch_loss)
        
        correct = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({X:mnist.test.images.reshape((-1, n_chunks, chunk_size)) , y:mnist.test.labels}))

train_neural_network(X)

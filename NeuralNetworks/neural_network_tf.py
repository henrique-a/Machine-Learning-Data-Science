import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500 # Number of nodes of hidden layer 1
n_nodes_hl2 = 500 # Number of nodes of hidden layer 2
n_nodes_hl3 = 500 # Number of nodes of hidden layer 3

n_classes = 10
batch_size = 100  

X = tf.placeholder(tf.float32, [None, 784]) # Defines type and shape of tensor (28px * 28px)
y = tf.placeholder(tf.float32) # Defines type and shape of tensor

def neural_network_model(X):
    hidden1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
    'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    
    hidden3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
    'biases': tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(X, hidden1_layer['weights']), hidden1_layer['biases'])
    l1 = tf.nn.relu(l1) # Activation function

    l2 = tf.add(tf.matmul(l1, hidden2_layer['weights']), hidden2_layer['biases'])
    l2 = tf.nn.relu(l2) # Activation function

    l3 = tf.add(tf.matmul(l2, hidden3_layer['weights']), hidden3_layer['biases'])
    l3 = tf.nn.relu(l3) # Activation function

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])
    
    return output

def train_neural_network(X):
    pred = neural_network_model(X)
    # Compute the mean of probability error measures in discrete classification
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdadeltaOptimizer().minimize(cost)

    n_epochs = 10 # Number of cycles (feed foward + backpropagation)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(n_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                # Get the next batch of entries
                epoch_X, epoch_y = mnist.train.next_batch(batch_size)
                # Train the nn with the batch of entries
                _, c = sess.run([optimizer, cost], feed_dict={X: epoch_X, y: epoch_y})
                epoch_loss += c
            
            print('Epoch', epoch, 'completed out of', n_epochs, 'loss', epoch_loss)
        
        correct = tf.equal(tf.arg_max(pred, 1), tf.arg_max(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy', accuracy.eval({X:mnist.test.images, y:mnist.test.labels}))

train_neural_network(X)

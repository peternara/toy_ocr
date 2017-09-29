'''
    Simple wrapper to fetch MNIST dataset and concat images to obtain sequences of numbers.
'''

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

width, height = 28, 28
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def get_mnist():
    train_labels, train_data = mnist.train.labels, mnist.train.images
    val_labels, val_data = mnist.validation.labels, mnist.validation.images

    train_data = np.reshape(train_data, (-1, 28, 28))
    train_labels = np.reshape(train_labels, (-1,5,10))
    val_data = np.reshape(val_data, (-1, 28, 28))
    val_labels = np.reshape(val_labels, (-1,5,10))

    train_data = np.transpose(train_data, (1,0,2))
    train_data = np.reshape(train_data, (28, -1, 28 * 5))
    train_data = np.transpose(train_data, (1,0,2))
    train_data = np.expand_dims(train_data, 3)

    val_data = np.transpose(val_data, (1,0,2))
    val_data = np.reshape(val_data, (28, -1, 28 * 5))
    val_data = np.transpose(val_data, (1,0,2))
    val_data = np.expand_dims(val_data, 3)
    
    return train_data, train_labels, val_data, val_labels


'''
    im2latex WIP, Jan Ivanecky
    MIT license
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
# load dataset, for now MNIST
train_data, train_labels, val_data, val_labels = get_mnist()

# functions for fetching variables
def get_weight_matrix(name, shape):
    return tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

def get_weight_matrix_conv2d(name, shape):
    return tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())

def get_bias_vector(name, shape):
    return tf.get_variable(name, shape, dtype=tf.float32, initializer=tf.constant_initializer(0.1))

# wrapper for convolution + relu
def convolutional_layer(name, channels, input):
    print(input.get_shape().as_list())
    input_channels = input.get_shape().as_list()[3]
    conv_filter = get_weight_matrix_conv2d(name + '_w', [3,3,input_channels,channels])
    conv_bias = get_bias_vector(name + '_b', [channels])
    conv = tf.nn.conv2d(input, conv_filter, [1,1,1,1], 'SAME')
    relu = tf.nn.relu(conv + conv_bias)
    return relu

# max pooling wrapper
def max_pooling(input):
    max = tf.nn.max_pool(input, [1,2,2,1], [1,2,2,1], 'SAME')
    return max

# I'm using MLP in multiple places so this comes in handy
def MLP(input, size, depth, output_size):
    x = input
    for i in range(depth):
        i_size = x.get_shape().as_list()[1] if i == 0 else size
        o_size = output_size if i == depth - 1 else size
        W = get_weight_matrix('W' + str(i), [i_size,o_size])
        b = get_bias_vector('b' + str(i), o_size)
        o = tf.matmul(x,W) + b
        x = tf.nn.relu(o) if i < depth - 1 else o
    return x

def ConvNet(input):

    '''
        Processes input image, outputs 'annotations'
        Note that it's fully convolutional so it takes image of any size as an input
    '''
    #print(input)
    relu1 = convolutional_layer('conv1', 32, input)
    relu1_2 = convolutional_layer('conv1_2', 32, relu1)
    max1 = max_pooling(relu1_2)
    relu2 = convolutional_layer('conv2', 64, max1)
    max2 = max_pooling(relu2)
    relu3 = convolutional_layer('conv3', 128, max2)
    relu4 = convolutional_layer('conv4', 256 , relu3)
    max4 = max_pooling(relu4)
    relu5 = convolutional_layer('conv5', 512 , max4)
    relu6 = convolutional_layer('conv6', 512 , relu5)
    max5 = max_pooling(relu6)
    dims = tf.shape(max5)
    width, height, channels = dims[2], dims[1], max5.get_shape().as_list()[3]
    output = tf.reshape(max5, [-1, height * width, channels])
    return output

def annotation_weights(annotations, state, scope):

    '''
        Computes weights for all annotations conditioned on the current state of LSTM
    '''

    annotation_count, feature_count = tf.shape(annotations)[1], annotations.get_shape().as_list()[2]
    state_size = state.get_shape().as_list()[1]
    c = annotations
    c = tf.reshape(c, [-1,feature_count])
    h = tf.expand_dims(state, 1)
    h = tf.tile(h, [1,annotation_count,1])
    h = tf.reshape(h, [-1, state_size])
    with tf.variable_scope(scope or 'attention') as scope_:
        y3 = MLP(tf.concat(1,[h,c]), state_size, 3, 1)
    y3 = tf.reshape(y3, [-1, annotation_count])
    e = tf.nn.softmax(y3)
    e = tf.expand_dims(e, 2)
    return e

class LSTM_Attention(tf.nn.rnn_cell.RNNCell):

    '''
        Implements LSTM cell with attention mechanism and deep output layer, it's basically just
        a wrapper for an LSTM cell.
        
        Note that input to the LSTM ('input' argument to the __call__ method) is not used at all, 
        because when attention mechanism is used, input for each time step is just a weighted sum
        of annotation vectors, where only the weights differ between time steps. I store 
        annotation vectors in the __init__ and compute weights when necessary.
        Other important thing to mention is that LSTM step is conditioned on the previous output, which
        means I need to store it with the LSTM state to be able to use this cell in combination with dynamic_rnn.
        I pack the output with the new state when returning the new state and unpack it at the begginning of the __call__. 
    '''

    def __init__(self, size, keep_rate, annotations, output_size):
        self.hidden_size = size
        self.out_size = output_size
        self.lstm = tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True)
        self.lstm = tf.nn.rnn_cell.DropoutWrapper(self.lstm, output_keep_prob=keep_rate)
        self.annotations = annotations
        #lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * depth, state_is_tuple=True)
        
    def init_state(self):
        mean_annotation = tf.reduce_mean(self.annotations, 1)
        with tf.variable_scope('init_c') as scope_c:
            init_c = MLP(mean_annotation, 128, 2, self.hidden_size)
        with tf.variable_scope('init_h') as scope_h:
            init_h = MLP(mean_annotation, 128, 2, self.hidden_size)
        return [tf.nn.rnn_cell.LSTMStateTuple(init_c, init_h), tf.zeros([tf.shape(self.annotations)[0], self.out_size])]

    @property
    def state_size(self):
        return self.hidden_size

    @property
    def output_size(self):
        return self.out_size

    def __call__(self, input, state, scope=None, prev_output=None):
        state, prev_output = state
        with tf.variable_scope(scope or type(self).__name__):  
            # compute current context
            with tf.variable_scope('attention') as scope:
                aw = annotation_weights(self.annotations, state[1], scope)
            current_context = tf.reduce_sum(aw * self.annotations, 1)
            
            # new state, conditioned on previous output, previous state and context
            input = tf.concat(1, [current_context, prev_output])
            with tf.variable_scope('decoder') as scope:
                output, state = self.lstm(input,state)

            # compute output of the deep output layer
            deep_output_input = tf.concat(1,[output, current_context, prev_output]) # output layer input is an LSTM input + current context + previous output
            with tf.variable_scope('output') as scope:
                deep_output = MLP(deep_output_input, 256, 1, self.out_size)
            output = tf.nn.softmax(deep_output)
        return output, [state, output] # packing state and output together to be able to use dynamic_rnn

# build the whole system, note that input image size can be arbitrary (should be constant within a batch ofc.)
#def build_model(vocabulary_size):
vocabulary_size = train_labels.shape[2] # 10
inputs = tf.placeholder(tf.float32, [None, None, None, 1])
labels = tf.placeholder(tf.float32, [None, None, vocabulary_size])
labels_ = tf.reshape(labels, [-1, vocabulary_size])
keep_rate = tf.placeholder(tf.float32)

# get annotation vectors
conv_output = ConvNet(inputs)

# compute prediction
lstm = LSTM_Attention(512, keep_rate, conv_output, vocabulary_size)
outputs, _ = tf.nn.dynamic_rnn(lstm, tf.zeros_like(labels), initial_state = lstm.init_state()) 
outputs = tf.reshape(outputs, [-1, vocabulary_size])

# accuracy and loss computation
correct_predictions = tf.equal(tf.cast(tf.argmax(outputs, 1), tf.int32), tf.cast(tf.argmax(labels_, 1), tf.int32))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
loss = tf.reduce_mean(-tf.reduce_sum(tf.log(outputs) * labels_,reduction_indices=[1]))

train = tf.train.AdamOptimizer(1e-4).minimize(loss)

#prediction = tf.argmax(outputs, 1)
#    return inputs, labels, train, accuracy, loss, keep_rate, outputs

#inputs, labels, train, accuracy, loss, keep_rate, outputs = build_model(train_labels.shape[2])

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

p = 0
BATCH_SIZE = 10
EVAL_INTERVAL = 1000
MAX_ITER = 100

for i in range(MAX_ITER + 1):
    nums = train_data[p: p + BATCH_SIZE]
    label = train_labels[p: p + BATCH_SIZE]
    p += BATCH_SIZE
    if p >= len(train_labels):
        p = 0

    feed_dict = {inputs: nums, labels: label, keep_rate: 0.5}
    l, _ = sess.run([loss, train], feed_dict)
    if i % 100 == 0:
        print('iteration: {}; train loss: {}'.format(i, l))
        
    if i % EVAL_INTERVAL == 0 and i > 0:
        feed_dict = {inputs: val_data, keep_rate: 1.0, labels: val_labels}
        acc = sess.run(accuracy, feed_dict)
        print('validation accuracy: {}'.format(acc))


result = outputs.eval(feed_dict={inputs: val_data, keep_rate: 1.0, labels: val_labels}, session=sess)
print(result)

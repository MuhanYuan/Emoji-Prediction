from __future__ import print_function
import re
import random
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

def merge_list(l):
    res = []
    for i in l:
        res += i
    return res

def sample_data(l, p=0.8):
    train = []
    test = []
    for i in l:
        if random.random()<0.8:
            train.append(i)
        else:
            test.append(i)
    return train, test

def padding(data_list, padding_len, ele_length):
    out_list = []
    for line in data_list:
        if len(line) == padding_len:
            out_list.append(line)
        elif len(line) < padding_len:
            out_list.append(line+[[0]*ele_length]*(padding_len - len(line)))
        else:
            out_list.append(line[:padding_len])
    return out_list

def vector_expanding(value, length):
    out_vector = [0]*length
    out_vector[value] = 1
    return out_vector

class TweetSplit:

    def __init__(self):
        self.tweet_list = None
        self.n_of_tweets = 0
        self.training_set_ratio = 0.6
        self.training_data = None
        self.testing_data = None
        self.validation_data = None
        self.emoji_list = []


    def read_data(self, file_name):
    	tw = []
    	with open(file_name, 'rb') as f:
    		for line in f:
    			data = line.decode('unicode-escape').split('\t')
    			tweet = self.pre_processing(data[0])
    			emoji = data[1].strip()
    			tw.append((tweet, emoji))
    			if emoji not in self.emoji_list:
    				self.emoji_list.append(emoji)
    	self.tweet_list = tw
    	self.n_of_tweets = len(self.tweet_list)

    def pre_processing(self, tweet):
    	# remove @name
    	tweet = re.sub(r"RT @\S+", "", tweet)
    	tweet = re.sub(r"@\S+", "", tweet)
    	#remove tag #string
    	tweet = re.sub(r"#\S+", "", tweet)
    	# remove url
    	tweet = re.sub(r"http\S+", "", tweet)
    	# replace all '\n' with space
    	tweet = tweet.replace('\n', ' ')
    	# remove emoji and punctuations
    	remove_list = set(re.findall(r'[^\w\s,]', tweet))
    	for ele in remove_list:
    		tweet = tweet.replace(ele, '')
    	tweet = tweet.lower().split(' ')
    	for i in range(len(tweet)-1, -1, -1):
    		if tweet[i] == '':
    			tweet = tweet[:i] + tweet[i+1:]

    	return ' '.join(tweet)

    def build_dataset(self):
        # define a word index dictionary
        word_list = set(merge_list([i[0].split() for i in self.tweet_list]))
        self.dictionary = dict()
        for word in word_list:
            self.dictionary[word] = len(self.dictionary)
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

        # define an emoji index dictionary
        emoji_list = set(merge_list([i[1].split() for i in self.tweet_list]))
        self.emoji_dic = dict()
        for emoji in emoji_list:
            self.emoji_dic[emoji] = len(self.emoji_dic)
        self.reverse_emoji_dic = dict(zip(self.emoji_dic.values(), self.emoji_dic.keys()))

        # sample training and testing data
        self.train_data, self.test_data = sample_data(self.tweet_list)



if __name__ == "__main__":
    myTS = TweetSplit()
    file_name = 'emoji_tweet_clean_1'
    myTS.read_data(file_name)
    myTS.build_dataset()

print ("start")

# Training Parameters
learning_rate = 0.01
training_steps = 1000
batch_size = 128
display_step = 10


# # Network Parameters
num_input = len(myTS.dictionary)
timesteps = 20
num_hidden = 16 # hidden layer num of features

num_classes = len(myTS.emoji_dic)

# # tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])


weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x, weights, biases):

    x = tf.unstack(x, timesteps, 1)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    test_data = np.array(padding([[ vector_expanding(myTS.dictionary[j], len(myTS.dictionary)) for j in i[0].split()] for i in myTS.test_data],20,len(myTS.dictionary)))
    test_label = np.array([vector_expanding(myTS.emoji_dic[j], len(myTS.emoji_dic)) for j in [i[1] for i in myTS.test_data]])
    for step in range(500):
        print ("loading")
        random.shuffle(myTS.train_data)
        train_data = myTS.train_data[:500]
        # print (train_data)
        x_train = np.array(padding([[ vector_expanding(myTS.dictionary[j], len(myTS.dictionary)) for j in i[0].split()] for i in train_data],20,len(myTS.dictionary)))
        y_train = np.array([vector_expanding(myTS.emoji_dic[j], len(myTS.emoji_dic)) for j in [i[1] for i in train_data]])
        # Run optimization op (backprop)
        print ("go")
        sess.run(train_op, feed_dict={X: x_train, Y: y_train})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: x_train,
                                                                 Y: y_train})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

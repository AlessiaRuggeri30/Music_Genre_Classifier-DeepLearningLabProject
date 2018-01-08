''' FEED-FORWARD NEURAL NETWORK'''


import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import random

# GENRE_TO_CLASSES = {
#     "classic pop and rock": 0,
#     "classical": 1,
#     "dance and electronica": 2,
#     "folk": 3,
#     "hip-hop": 4,
#     "jazz and blues": 5,
#     "metal": 6,
#     "pop": 7,
#     "punk": 8,
#     "soul and reggae": 9
# }

GENRE_TO_CLASSES = {
	"classic pop and rock": 0,
	"dance and electronica": 1,
	"jazz and blues": 2,
	"punk": 3,
}


def manage_dataset(dataset):
	''' Function that divides the dataset into inputs and targets with the right shape,
		it also shuffles the data '''
	x = np.array([x[0] for x in dataset])
	y = np.array([[y[1] for y in dataset]])
	y = y.reshape([-1, 1])

	c = list(zip(x, y))
	random.shuffle(c)
	x_random, y_random = map(list, zip(*c))
	x_random = np.array([x for x in x_random])
	y_random = np.array([y for y in y_random])

	return x_random, y_random


''' Prepare dataset inputs and targets '''
print("---------- Loading data... ----------")
# database = trainingset = np.load("./data/dataset/trainingset_array.npy")
# dataset = validationset = np.load("./data/dataset/validationset_array.npy")
# dataset = testset = np.load("./data/dataset/testset_array.npy")
database = trainingset = np.load("./data/dataset4c_np/trainingset_np.npy")

train_x, train_y = manage_dataset(database)


''' Parameters '''
# hyperparameters
learning_rate = 0.001
epochs = 50
batch_size = 64
n_batches = len(database) // batch_size
print("Number of batches for each epoch:", n_batches)

# network parameters
n_samples = trainingset.shape[0]
n_input = 36  # input size
n_classes = 4
layers_dim = np.array([64, 128, 128, n_classes])
n_layers = len(layers_dim)
dropout = 0.80


''' Variables '''
# neural network variables
x = tf.placeholder(tf.float32, [None, n_input], name='x')
y = tf.placeholder(tf.int64, [None, n_classes], name='y')
# y = tf.one_hot(y, depth=n_classes, name='y_hot')

keep_prob = tf.placeholder(tf.float32)

# dataset variables
# dataset_x = tf.placeholder(tf.float32, [None, n_input])
# dataset_y = tf.placeholder(tf.int64, [None, n_classes])
#
# dataset = tf.data.Dataset().from_tensor_slices((x, y)).batch(batch_size).repeat()
# iterator = dataset.make_initializable_iterator()


def getBatch(x, y, batch_size, iteration):
	''' Function that returns the next bach of data to be computed '''
	start_b = (iteration * batch_size) % len(x)
	end_b = ((iteration * batch_size) + batch_size) % len(x)

	if start_b < end_b:
		return x[start_b:end_b], y[start_b:end_b]
	else:
		batch_x = np.vstack((x[start_b:], x[:end_b]))
		batch_y = np.vstack((y[start_b:], y[:end_b]))

		return batch_x, batch_y


def one_hot_encoder(y):
	''' Function that applies one hot encoding to targets '''
	onehot = list()
	for value in y:
		letter = [0 for _ in range(n_classes)]
		letter[value[0]] = 1
		onehot.append(letter)
	onehot = np.array(onehot)
	return onehot


''' Feed-Forward functions'''


def create_layer(input, n_size, activation=None):
	''' Function that returns a layer of neurons with the given size and activation'''
	# create weights and biases
	size_input = input.get_shape().as_list()[-1]
	W = tf.Variable(tf.truncated_normal(shape=[size_input, n_size], stddev=0.1))
	b = tf.Variable(tf.zeros([n_size]))

	# calculate output
	output = tf.matmul(input, W) + b

	if activation is not None:
		output = activation(output)
	return output


def fc_neural_network(input):
	''' Function that uses create_layer to generate a feed-forward neural network;
		Only the last layer is activated'''
	for i in range(n_layers):
		if i == (n_layers - 1):
			nonactived = create_layer(input, layers_dim[i])
			input = create_layer(input, layers_dim[i], tf.nn.softmax)
		else:       # tf.nn.tanh or tf.nn.relu
			input = tf.layers.dropout(create_layer(input, layers_dim[i], tf.nn.tanh), rate=dropout)

	return input, nonactived


'''Perform training'''


def plot_results(title, tot_loss, tot_acc, y_lim=True):
	plt.close()
	lim = 10 if y_lim else max(tot_loss)+1
	y_step = 0.5

	plt.suptitle(title,
				 fontsize=14, fontweight="bold")
	plt.title("learning_rate = {},   epochs = {},   batch_size = {},   n_classes = {},   dropout = {}"
			  .format(learning_rate, epochs, batch_size, n_classes, dropout),
			  fontsize=10)
	plt.xlabel("Epochs")
	plt.ylabel("Values")
	plt.plot(tot_loss, label="loss")
	plt.plot(tot_acc, label="accuracy")
	plt.xlim(0, epochs)
	plt.ylim(0, lim)
	plt.axhline(y=1, c="lightgrey", linewidth=0.7, zorder=0)
	plt.xticks(np.arange(0, epochs, 10))
	plt.yticks(np.arange(0, lim, y_step))
	plt.legend()
	plt.show()


def model_training():
	''' Function that performs the training of the neural network '''
	output, output_nonactived = fc_neural_network(x)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_nonactived, labels=y))
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	correctness = tf.equal(tf.argmax(output, -1), tf.argmax(y, -1))
	accuracy = tf.reduce_mean(tf.cast(correctness, 'float'))

	# Initialize a session
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()
	# sess.run(tf.one_hot(y, depth=n_classes), feed_dict={y: train_y})
	# sess.run(iterator.initializer, feed_dict={x: train_x, 'y:0': train_y})

	tot_loss = []
	tot_acc = []

	for epoch in range(epochs):
		avg_loss = 0
		avg_acc = 0

		for i in range(n_batches):
			batch_x, batch_y = getBatch(train_x, train_y, batch_size, i)
			# batch_x, batch_y = sess.run(iterator.get_next())
			_, loss_value, acc = sess.run([train_step, loss, accuracy], feed_dict={x: batch_x, y: batch_y})
			avg_loss += loss_value
			avg_acc += acc

		avg_loss = avg_loss / n_batches
		avg_acc = avg_acc / n_batches
		tot_loss.append(avg_loss)
		tot_acc.append(avg_acc)
		print("---------")
		print("Epoch: {}\n  AVG loss: {:.5f}\n  AVG acc: {:.5f}".format(epoch+1, avg_loss, avg_acc))

	print("FINISHED!")
	print()
	print("---------- Saving model... ----------")
	save_path = saver.save(sess, "models/fc_model/fc_model.ckpt")
	print("Model saved in file: %s" % save_path)
	print()
	print("---------- Plotting results... ----------")
	plot_results("Feed-Forward Neural Network", tot_loss, tot_acc, y_lim=False)


tf.set_random_seed(0)
train_y = one_hot_encoder(train_y)
model_training()

'''Just after sess.run(tf.global_variables_initializer())'''
# saver.restore(sess, "models/fc_model/fc_model.ckpt")
# print("Model restored.")

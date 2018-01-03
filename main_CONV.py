import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import random

# GENRE_TO_CLASSES = {
# 	"classic pop and rock": 0,
# 	"classical": 1,
# 	"dance and electronica": 2,
# 	"folk": 3,
# 	"hip-hop": 4,
# 	"jazz and blues": 5,
# 	"metal": 6,
# 	"pop": 7,
# 	"punk": 8,
# 	"soul and reggae": 9
# }


GENRE_TO_CLASSES = {
	"classic pop and rock": 0,
	"dance and electronica": 1,
	"jazz and blues": 2,
	"punk": 3,
}


def manage_dataset(dataset):
	''' Function that divide the dataset into inputs and targets with the right shape,
		it also shuffle the data '''
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
# database = trainingset = np.load("./data/subset_np/trainingset_np.npy")
# dataset = validationset = np.load("./data/subset_np/validationset_np.npy")
# dataset = testset = np.load("./data/subset_np/testset_np.npy")
database = trainingset = np.load("./data/subset4c_np/trainingset_np.npy")

train_x, train_y = manage_dataset(database)

''' Parameters '''
# hyperparameters
learning_rate = 0.001
epochs = 10
batch_size = 1
n_batches = len(database) // batch_size
print("Number of batches for each epoch:", n_batches)

# network parameters
n_samples = database.shape[0]
n_seg = train_x[0].shape[0]  # 5468
n_coef = train_x[0].shape[1]  # 12
n_classes = 4
layers_dim = np.array([64, 128, 64, n_classes])
n_layers = len(layers_dim)
dropout = 0.8
print("Number of segments for each song:", n_seg)

''' Variables '''
x = tf.placeholder(tf.float32, [None, n_seg, n_coef])
y = tf.placeholder(tf.int64, [None, n_classes])

keep_prob = tf.placeholder(tf.float32)


def getBatch(x, y, batch_size, iteration):
	''' Function that return the next bach of data to be computed '''
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


''' Convolutional functions'''


def conv2d(x, W):
	''' Function that return a 2-dimension convolution '''
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
	''' Function that return a 2-dimension max-pooling '''
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
	''' Function that return random weights of given shape '''
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	''' Function that return random biases of given shape '''
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def create_conv_layer(x, n_channel, filter_size, n_filters):
	""" Function that return a convolutional layer of neurons
		with the given number of filters, having the given shape
		and number of channels """
	shape = [filter_size[0], filter_size[1], n_channel, n_filters]

	W = weight_variable(shape)
	b = bias_variable([n_filters])

	layer = tf.nn.relu(conv2d(x, W) + b)
	layer = maxpool2d(layer)

	return layer


def create_fc_layer(x, shape):
	''' Function that return a feed-forward layer of activated neurons
		with the given size'''
	W = weight_variable(shape)
	b = bias_variable([shape[1]])

	layer = tf.matmul(x, W) + b
	layer = tf.nn.relu(layer)

	return layer


def convolutional_neural_network(x):
	''' Function that uses create_conv_layer and create_fc_layer to generate
		a convolutional neural network '''
	x = tf.reshape(x, shape=[-1, n_seg, n_coef, 1])  # n_batches, (db.shape), db.n_channels

	conv1 = create_conv_layer(x, 1, (128, 4), 16)

	conv2 = create_conv_layer(conv1, 16, (128, 4), 32)
	# print(conv2.shape)
	conv2_shape = int(conv2.shape[1])


	fc = tf.reshape(conv2, [-1, conv2_shape * 3 * 32])  # conv2.shape
	fc = create_fc_layer(fc, [conv2_shape * 3 * 32, 512])  # conv2.shape, num of neurons in the fc layer
	fc = tf.nn.dropout(fc, dropout)

	output = create_fc_layer(fc, [512, n_classes])	#num of neurons in the fc layer, n_classes

	return output


''' Perform training '''


def plot_results(title, tot_loss, tot_acc, y_lim=True):
	plt.close()
	lim = 10 if y_lim else max(tot_loss)+1
	y_step = 0.5 if y_lim else 10

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
	plt.xticks(np.arange(0, epochs, 1.0))
	plt.yticks(np.arange(0, lim, y_step))
	plt.legend()
	plt.show()


def reverse_dic(ind):
	genre = list(GENRE_TO_CLASSES.keys())[list(GENRE_TO_CLASSES.values()).index(ind)]
	return genre


def model_training():
	''' Function that performs the training of the neural network '''
	output = convolutional_neural_network(x)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	correctness = tf.equal(tf.argmax(output, -1), tf.argmax(y, -1))
	accuracy = tf.reduce_mean(tf.cast(correctness, 'float'))

	# Initialize a session
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	tot_loss = []
	tot_acc = []

	for epoch in range(epochs):
		print("---------")
		print("Computing epoch {} of {}".format(epoch, epochs))
		genres_acc = []
		avg_loss = 0
		avg_acc = 0
		right = [0] * n_classes
		all = [0] * n_classes

		for i in range(n_batches):
			print("\tComputing batch {} of {}".format(i, n_batches))
			batch_x, batch_y = getBatch(train_x, train_y, batch_size, i)
			# _, loss_value, acc = sess.run([train_step, loss, accuracy],
			# 							  feed_dict={x: batch_x, y: batch_y})

			if batch_size == 1:
				out, _, loss_value, acc = sess.run([output, train_step, loss, accuracy],
												   feed_dict={x: batch_x, y: batch_y})
				out = np.argmax(out)
				target = np.argmax(batch_y)
				if out == target:
					right[target] += 1
				all[target] += 1

			else:
				_, loss_value, acc = sess.run([train_step, loss, accuracy],
											  feed_dict={x: batch_x, y: batch_y})

			if i % 100 == 0:
				print("\tloss: ", loss_value)
				# print("\tacc: ", acc)
			avg_loss += loss_value
			avg_acc += acc

		avg_loss = avg_loss / n_batches
		avg_acc = avg_acc / n_batches
		tot_loss.append(avg_loss)
		tot_acc.append(avg_acc)
		print("----- Epoch: {}\n  AVG Loss: {:.5f}\n  AVG acc: {:.5f}".format(epoch, avg_loss, avg_acc))
		if batch_size == 1:
			for i in range(n_classes):
				genres_acc.append(right[i] / all[i] if all[i] != 0 else 0)
			for i in range(n_classes):
				print("   ", reverse_dic(i), ": {:.4f}".format(genres_acc[i]))
		print()

	print("FINISHED!")
	print()
	print("---------- Saving model... ----------")
	save_path = saver.save(sess, "models/conv_model/conv_model.ckpt")
	print("Model saved in file: %s" % save_path)
	print()
	print("---------- Plotting results... ----------")
	plot_results("Convolutional Neural Network", tot_loss, tot_acc)
	plot_results("Convolutional Neural Network", tot_loss, tot_acc, y_lim=False)


tf.set_random_seed(0)
train_y = one_hot_encoder(train_y)
model_training()

''' CONVOLUTIONAL NEURAL NETWORK'''

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


def shuffle_data(x, y):
	''' Function that shuffles the data '''
	c = list(zip(x, y))
	random.shuffle(c)
	x_random, y_random = map(list, zip(*c))
	x_random = np.array([x for x in x_random])
	y_random = np.array([y for y in y_random])

	return x_random, y_random


def manage_dataset(dataset):
	''' Function that divides the dataset into inputs and targets with the right shape,
		it also shuffles the data '''
	x = np.array([x[0] for x in dataset])
	y = np.array([[y[1] for y in dataset]])
	y = y.reshape([-1, 1])

	x_random, y_random = shuffle_data(x, y)

	return x_random, y_random


''' Prepare dataset inputs and targets '''
print("---------- Loading data... ----------")
# database = trainingset = np.load("./data/subset_np/trainingset_np.npy")
# dataset = validationset = np.load("./data/subset_np/validationset_np.npy")
# dataset = testset = np.load("./data/subset_np/testset_np.npy")
database = trainingset = np.load("./data/subset4c_np/trainingset_np.npy")
validationset = np.load("./data/subset4c_np/validationsubset_np.npy")
testset = np.load("./data/subset4c_np/testset_np.npy")

train_x, train_y = manage_dataset(database)
val_x, val_y = manage_dataset(validationset)
test_x, test_y = manage_dataset(testset)

''' Parameters '''
# hyperparameters
learning_rate = 0.001
epochs = 10
batch_size = 32
n_batches = len(database) // batch_size
print("Number of batches for each epoch:", n_batches)

# network parameters
n_samples = database.shape[0]
n_seg = train_x[0].shape[0]
n_coef = train_x[0].shape[1]  # 12
n_classes = 4
dropout = 0.5
print("Number of segments for each song:", n_seg)

''' Variables '''
x = tf.placeholder(tf.float32, [None, n_seg, n_coef], name='x')
y = tf.placeholder(tf.int64, [None, n_classes], name='y')

keep_prob = tf.placeholder(tf.float32)


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


''' Convolutional functions'''


def conv2d(x, W):
	''' Function that returns a 2-dimension convolution '''
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
	''' Function that returns a 2-dimension max-pooling '''
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
	''' Function that returns random weights of given shape '''
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	''' Function that returns random biases of given shape '''
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def create_conv_layer(x, n_channel, filter_size, n_filters):
	''' Function that returns a convolutional layer of neurons
		with the given number of filters, having the given shape
		and number of channels '''
	shape = [filter_size[0], filter_size[1], n_channel, n_filters]

	W = weight_variable(shape)
	b = bias_variable([n_filters])

	layer = tf.nn.relu(conv2d(x, W) + b)
	layer = maxpool2d(layer)

	return layer


def create_fc_layer(x, shape, last=False):
	''' Function that returns a feed-forward layer of activated neurons
		with the given size'''
	W = weight_variable(shape)
	b = bias_variable([shape[1]])

	layer = tf.matmul(x, W) + b

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
	fc = tf.nn.relu(fc)
	fc = tf.nn.dropout(fc, dropout)

	output = create_fc_layer(fc, [512, n_classes], last=True)	#num of neurons in the fc layer, n_classes
	output_act = tf.nn.softmax(output)

	return output, output_act


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


def class_accuracy(out, batch_y, right_out, all_out):
	for i in range(batch_size):
		out_i = np.argmax(out[i])
		target_i = np.argmax(batch_y[i])
		if out_i == target_i:
			right_out[target_i] += 1
		all_out[target_i] += 1


def model_training(train_x, train_y, restore_model=False):
	''' Function that performs the training of the neural network '''
	output_nonactivated, output = convolutional_neural_network(x)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_nonactivated, labels=y))
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

	correctness = tf.equal(tf.argmax(output, -1), tf.argmax(y, -1))
	accuracy = tf.reduce_mean(tf.cast(correctness, 'float'))

	# Initialize a session
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	tot_loss = []
	tot_acc = []

	if (restore_model):
		print("---------- Restoring model... ----------")
		saver.restore(sess, "models/conv_model3_mid/conv_model3_mid.ckpt")

		print("---------- Computing outputs... ----------")
		loss = loss.eval(feed_dict={x: test_x, y: test_y})
		acc = accuracy.eval(feed_dict={x: test_x, y: test_y})
		print()
		print("----- Results:\n  Loss: {:.5f}\n  Acc: {:.5f}".format(loss, acc))

		print("FINISHED!")

	else:
		with open('results/conv4_results.txt', 'w') as f:
			f.write("Number of batches for each epoch: {}\n".format(n_batches))
			f.write("Number of segments for each song: {}\n".format(n_seg))

			for epoch in range(epochs):
				if epoch == 7:
					print("---------- Saving model... ----------")
					save_path = saver.save(sess, "models/conv_model4_mid/conv_model4_mid.ckpt")

				print("---------")
				print("Computing epoch {} of {}".format(epoch, epochs))

				f.write("---------\n")
				f.write("Computing epoch {} of {}\n".format(epoch, epochs))

				genres_acc = []
				avg_loss = 0
				avg_acc = 0
				avg_loss_val = 0
				avg_acc_val = 0
				right_out = [0] * n_classes
				all_out = [0] * n_classes

				for i in range(n_batches):
					if i % 5 == 0:
						print("\tComputing batch {} of {}".format(i, n_batches))
					batch_x, batch_y = getBatch(train_x, train_y, batch_size, i)

					out, _, loss_value, acc = sess.run([output, train_step, loss, accuracy],
													   feed_dict={x: batch_x, y: batch_y})

					class_accuracy(out, batch_y, right_out, all_out)

					loss_val, acc_val = sess.run([loss, accuracy],
												 feed_dict={x: val_x, y: val_y})

					if i % 5 == 0:
						print("\tloss: ", loss_value)
						print("\tacc: ", acc)
						print("\tloss validation: ", loss_val)
						print("\tacc validation: ", acc_val)
						print("\t---------")

						f.write("\tComputing batch {} of {}\n".format(i, n_batches))
						f.write("\tloss: {}\n".format(loss_value))
						f.write("\tacc: {}\n".format(acc))
						f.write("\tloss validation: {}\n".format(loss_val))
						f.write("\tacc validation: {}\n".format(acc_val))
						f.write("\t---------\n")

					avg_loss += loss_value
					avg_acc += acc
					avg_loss_val += loss_val
					avg_acc_val += acc_val

				avg_loss = avg_loss / n_batches
				avg_acc = avg_acc / n_batches
				tot_loss.append(avg_loss)
				tot_acc.append(avg_acc)

				avg_loss_val = avg_loss_val / n_batches
				avg_acc_val = avg_acc_val / n_batches

				print("----- Epoch: {}\n  AVG loss: {:.5f}\n  AVG acc: {:.5f}".format(epoch, avg_loss, avg_acc))
				f.write("----- Epoch: {}\n  AVG loss: {:.5f}\n  AVG acc: {:.5f}\n".format(epoch, avg_loss, avg_acc))

				for i in range(n_classes):
					genres_acc.append(right_out[i] / all_out[i] if all_out[i] != 0 else 0)
				for i in range(n_classes):
					print("   ", reverse_dic(i), ": {:.4f}".format(genres_acc[i]))
					f.write("   {}: {:.4f}\n".format(reverse_dic(i), genres_acc[i]))

				print("  AVG loss validation: {:.5f}\n  AVG acc validation: {:.5f}".format(epoch, avg_loss_val, avg_acc_val))
				f.write("  AVG loss validation: {:.5f}\n  AVG acc validation: {:.5f}\n".format(epoch, avg_loss_val,
																							avg_acc_val))
				print()
				f.write("\n")

			print("---------- Computing test outputs... ----------")
			loss_test, acc_test = sess.run([loss, accuracy], feed_dict={x: test_x, y: test_y})
			print("----- Test results:\n  Loss: {:.5f}\n  Acc: {:.5f}".format(loss_test, acc_test))
			print()
			print("FINISHED!")
			print()
			print("---------- Saving model... ----------")
			save_path = saver.save(sess, "models/conv_model4/conv_model4.ckpt")
			print("Model saved in file: %s" % save_path)
			print()
			print("---------- Plotting results... ----------")
			plot_results("Convolutional Neural Network", tot_loss, tot_acc)
			plot_results("Convolutional Neural Network", tot_loss, tot_acc, y_lim=False)

			f.write("\n")
			f.write("---------- Computing test outputs... ----------\n")
			f.write("----- Test results:\n  Loss: {:.5f}\n  Acc: {:.5f}".format(loss_test, acc_test))
			f.write("\n")
			f.write("FINISHED!")


tf.set_random_seed(0)
train_y = one_hot_encoder(train_y)
val_y = one_hot_encoder(val_y)
test_y = one_hot_encoder(test_y)
model_training(train_x, train_y, restore_model=False)

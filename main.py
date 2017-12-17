import numpy as np
import math
import tensorflow as tf
import random

GENRE_TO_CLASSES = {
    "classic pop and rock": 0,
    "classical": 1,
    "dance and electronica": 2,
    "folk": 3,
    "hip-hop": 4,
    "jazz and blues": 5,
    "metal": 6,
    "pop": 7,
    "punk": 8,
    "soul and reggae": 9
}


def manage_dataset(dataset):
    x = np.array([x[0] for x in dataset])
    y = np.array([[y[1] for y in dataset]])
    y = y.reshape([-1, 1])

    c = list(zip(x, y))
    random.shuffle(c)
    x_random, y_random = map(list, zip(*c))
    x_random = np.array([x for x in x_random])
    y_random = np.array([y for y in y_random])

    return x_random, y_random


'''Prepare dataset inputs and targets'''

database = trainingset = np.load("./dataset/trainingset_array.npy")
# dataset = validationset = np.load("./dataset/validationset_array.npy")
# dataset = testset = np.load("./dataset/testset_array.npy")

train_x, train_y = manage_dataset(database)


'''Parameters'''

# hyperparameters
learning_rate = 0.01
epochs = 50000
batch_size = 128
n_batches = len(database) // batch_size
print("Number of batches for each epoch:", n_batches)

# network parameters
n_samples = trainingset.shape[0]
n_input = 36  # input size
n_classes = 10
layers_dim = np.array([64, 128, 32, n_classes])
n_layers = len(layers_dim)
dropout = 0.80  # avoid overfitting (randomly turns off neurons, to force new path research)


'''Variables'''

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
    start_b = (iteration * batch_size) % len(x)
    end_b = ((iteration * batch_size) + batch_size) % len(x)

    if start_b < end_b:
        return x[start_b:end_b], y[start_b:end_b]
    else:
        batch_x = np.vstack((x[start_b:], x[:end_b]))
        batch_y = np.vstack((y[start_b:], y[:end_b]))

        return batch_x, batch_y


def one_hot_encoder(y):
    # one hot encode
    onehot = list()
    for value in y:
        letter = [0 for i in range(n_classes)]
        letter[value[0]] = 1
        onehot.append(letter)
    onehot = np.array(onehot)
    return onehot

def create_layer(input, n_size, activation=None):
    # create weights and biases
    size_input = input.get_shape().as_list()[-1]
    W = tf.Variable(tf.truncated_normal(shape=[size_input, n_size], stddev=0.1))
    b = tf.Variable(tf.zeros([n_size]))

    # calculate output
    output = tf.matmul(input, W) + b

    if activation is not None:
        output = activation(output)
    return output


def create_model(input):
    for i in range(n_layers):
        if i == (n_layers - 1):
            nonactived = create_layer(input, layers_dim[i])
            input = create_layer(input, layers_dim[i], tf.nn.softmax)
        else:       # tf.nn.tanh or tf.nn.relu
            input = tf.layers.dropout(create_layer(input, layers_dim[i], tf.nn.tanh), rate=dropout)

    return input, nonactived


'''Perform training'''


def model_training():
    output, output_nonactived = create_model(x)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_nonactived, labels=y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    correctness = tf.equal(tf.argmax(output, -1), tf.argmax(y, -1))
    accuracy = tf.reduce_mean(tf.cast(correctness, 'float'))

    # Initialize a session
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.one_hot(y, depth=n_classes), feed_dict={y: train_y})
    # sess.run(iterator.initializer, feed_dict={x: train_x, 'y:0': train_y})

    for epoch in range(epochs):
        avg_loss = 0
        avg_acc = 0

        for i in range(batch_size):
            batch_x, batch_y = getBatch(train_x, train_y, batch_size, i)
            # batch_x, batch_y = sess.run(iterator.get_next())
            _, loss_value, acc = sess.run([train_step, loss, accuracy], feed_dict={x: batch_x, y: batch_y})
            avg_loss += loss_value
            avg_acc += acc

        avg_loss = avg_loss / n_batches
        avg_acc = avg_acc / n_batches
        print("---------")
        print("Epoch: {}\n  AVG Loss: {:.5f}\n  AVG acc: {:.5f}".format(epoch, avg_loss, avg_acc))

    print("FINISHED!")


train_y = one_hot_encoder(train_y)
model_training()

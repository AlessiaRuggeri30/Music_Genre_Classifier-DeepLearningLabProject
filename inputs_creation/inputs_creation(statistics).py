import json
import glob
import numpy as np
import random
import time

GEN = "genre"
SEG_START = "segments_start"
MFCC = "segments_timbre"
SONG = "songs"

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


''' Source folder of the dataset '''
# file_directory = "../data/dataset/"
# file_directory = "../data/subset/"
file_directory = "../data/subset4c/"

n_classes = len(GENRE_TO_CLASSES)
percentages = [70, 20, 10]


''' Useless things '''
''' ------------------------------ '''

# start = time.process_time()

# print('Track id: {}'.format(data[SONG]["value"][0][30]))
# print('\tMusic genre: {}'.format(data[GEN]))
# print('\tList of segment starts: {}'.format(data[SEG_START]["value"]))
# print('\tList of MFCCs: {}'.format(data[MFCC]["value"]))
# print()

def normalize(x, minimum, maximum):
	normalized = (2 * (x - minimum) / (maximum - minimum)) - 1
	return normalized


def input_normalizer(m, sd, d):
	realmin = np.amin([np.amin(m), np.amin(sd), np.amin(d)])
	realmax = np.amax([np.amax(m), np.amax(sd), np.amax(d)])
	err = ((realmax - realmin) * 2) / 100
	newmin = realmin - err
	newmax = realmax + err

	m_norm = np.array(list())
	sd_norm = np.array(list())
	d_norm = np.array(list())

	for x in m:
		m_norm = np.append(m_norm, normalize(x, newmin, newmax))
	for x in sd:
		sd_norm = np.append(sd_norm, normalize(x, newmin, newmax))
	for x in d:
		d_norm = np.append(d_norm, normalize(x, newmin, newmax))

	return m_norm, sd_norm, d_norm


''' ------------------------------ '''


def MFCCs_manager(MFCCs):
	m = np.mean(MFCCs, axis=0)
	sd = np.std(MFCCs, axis=0)
	d = np.ptp(MFCCs, axis=0)
	return m, sd, d


def input_creator(MFCCs, genre):
	m, sd, d = MFCCs_manager(MFCCs)
	label = GENRE_TO_CLASSES[genre]
	values = np.append(m, [sd, d])
	sample = [values, label]
	return sample


def inputs_creator():
	files = glob.glob(file_directory + '/**/*.json', recursive=True)
	y = []

	for i, file in enumerate(files):
		print("doing {} of {}".format(i, len(files)))
		with open(file) as fl:
			json_data = fl.read()
			data = json.loads(json_data)

			MFCCs = np.asarray(data[MFCC]["value"])
			# n_seg = MFCCs.shape[0]
			# n_coef = MFCCs.shape[1]
			# print(n_seg, n_coef)
			genre = data[GEN]

			sample = input_creator(MFCCs, genre)

			y.append(sample)

	y = np.array(y)

	return y


def split_dataset(dataset):
	trainingset = []  # 70%
	validationset = []  # 20%
	testset = []  # 10%

	for i in range(n_classes):
		if i in (0, 2, 5, 8):
			print("splitting {} of {}".format(i, n_classes))
			items = dataset[dataset[:, 1] == i]
			class_len = len(items)

			perc_train = (class_len * percentages[0]) // 100
			perc_val = (class_len * percentages[1]) // 100
			# perc_test = class_len - (perc_train + perc_val)

			trainingset.append(items[:perc_train])
			validationset.append(items[perc_train:(perc_train + perc_val)])
			testset.append(items[(perc_train + perc_val):])

	training = []
	validation = []
	test = []
	for j in range(10):
		for k in range(len(trainingset[j])):
			training.append(trainingset[j][k])
		for h in range(len(validationset[j])):
			validation.append(validationset[j][h])
		for l in range(len(testset[j])):
			test.append(testset[j][l])

	trainingset = np.array(training)
	validationset = np.array(validation)
	testset = np.array(test)

	return trainingset, validationset, testset


''' Destination folder for the numpy file '''
# fd = "../data/dataset_np/"
# fd = "../data/subset_np/"
fd = "../data/subset4c_np/"


''' Input creator (create the array representing the whole dataset) '''
y = inputs_creator()

np.save(fd + 'dataset_np.npy', y)


''' Split dataset (split the whole dataset into training, validation and test set '''
trainingset, validationset, testset = split_dataset(y)
np.save(fd + 'trainingset_np.npy', trainingset)
np.save(fd + 'validationset_np.npy', validationset)
np.save(fd + 'testset_np.npy', testset)


''' Load dataset from numpy file '''
# print("---------- Loading data... ----------")
# trainingset = fd + "trainingset_array.npy"
# validationset = fd + "validationset_array.npy"
# testset = fd + "testset_array.npy"
#
# trainingset = np.load(trainingset)
# validationset = np.load(validationset)
# testset = np.load(testset)


''' Some test '''
# print(trainingset.shape)
# print(validationset.shape)
# print(testset.shape)

# train_x = np.array([x[0] for x in trainingset])[:3]
# train_y = np.array([[y[1] for y in trainingset]])
# train_y = train_y.reshape([-1, 1])[:3]

# print(train_x)
# print(train_y)

# c = list(zip(train_x, train_y))
# random.shuffle(c)
# train_x, train_y = map(list, zip(*c))
# train_x = np.array([x for x in train_x])
# train_y = np.array([y for y in train_y])

# print(train_x)
# print(train_y)



# end = time.process_time()

# print()
# print('\tTime needed: {}'.format(end - start))

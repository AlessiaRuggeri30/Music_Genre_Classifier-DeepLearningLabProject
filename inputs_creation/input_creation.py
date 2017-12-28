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
file_directory = "../data/subset/"


n_classes = len(GENRE_TO_CLASSES)
percentages = [70, 20, 10]

def inputs_creator():
	files = glob.glob(file_directory + '/**/*.json', recursive=True)
	y = []

	for i, file in enumerate(files):
		print("doing {} of {}".format(i, len(files)))
		with open(file) as fl:
			json_data = fl.read()
			data = json.loads(json_data)

			MFCCs = np.asarray(data[MFCC]["value"])
			n_seg = MFCCs.shape[0]
			n_coef = MFCCs.shape[1]
			# print(n_seg, n_coef)
			genre = data[GEN]
			label = GENRE_TO_CLASSES[genre]

			sample = np.array([MFCCs, label])

			y.append(sample)

		# if i == 3000:
		# 	break

	y = np.array(y)

	return y

def dataset_padding(y):
	max_num_fragment = np.max([(item[0].shape[0]+1) for item in y])
	max_shape = [max_num_fragment, 12]

	b = True

	for song in y:
		item = np.zeros((max_shape[0], max_shape[1]))
		item[:song[0].shape[0], :song[0].shape[1]] = song[0]

		song[0] = item


def split_dataset(dataset):
	trainingset = []  # 70%
	validationset = []  # 20%
	testset = []  # 10%

	for i in range(n_classes):
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
fd = "../data/subset_np/"


''' Input creator (create the array representing the whole dataset) '''
# y = inputs_creator()
# dataset_padding(y)

# np.save(fd + 'dataset_np.npy', y)


''' Split dataset (split the whole dataset into training, validation and test set '''
# trainingset, validationset, testset = split_dataset(y)
# np.save(fd + 'trainingset_np.npy', trainingset)
# np.save(fd + 'validationset_np.npy', validationset)
# np.save(fd + 'testset_np.npy', testset)


''' Load dataset from numpy file '''
db = np.load(fd + "trainingset_np.npy")
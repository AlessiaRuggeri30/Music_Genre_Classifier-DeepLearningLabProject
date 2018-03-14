import numpy as np

GENRE_TO_CLASSES = {
	"classic pop and rock": 0,
	"dance and electronica": 1,
	"jazz and blues": 2,
	"punk": 3,
}

file_directory = "../data/subset4c/"

n_classes = len(GENRE_TO_CLASSES)
statistics = False


def split_dataset(dataset):
	validationset = []  # 33%

	for i in range(n_classes):
		print("splitting {} of {}".format(i, n_classes))
		items = dataset[dataset[:, 1] == i]
		class_len = len(items)

		perc_val = (class_len * 33) // 100

		validationset.append(items[:perc_val])

	validation = []
	for j in range(4):
		for h in range(len(validationset[j])):
			validation.append(validationset[j][h])

	validationset = np.array(validation)

	return validationset


fd = "../data/subset4c_np/"

print("---------- Loading data... ----------")
db = np.load(fd + "validationset_np.npy")
print(db.shape)
# print("---------- Splitting data... ----------")
# validationset = split_dataset(db)
# print("---------- Saving validationsubset... ----------")
# np.save(fd + 'validationsubset_np.npy', validationset)

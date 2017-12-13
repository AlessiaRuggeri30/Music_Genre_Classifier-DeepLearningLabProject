import json
from pprint import pprint
import numpy as np
import time

file_directory = "../subtrainingset/TRAAABD128F429CF47.json"
json_data = open(file_directory).read()

GEN = "genre"
SEG_START = "segments_start"
MFCC = "segments_timbre"
SONG = "songs"

start = time.process_time()

data = json.loads(json_data)

print('Track id: {}'.format(data[SONG]["value"][0][30]))
print('\tMusic genre: {}'.format(data[GEN]))
print('\tList of segment starts: {}'.format(data[SEG_START]["value"]))
print('\tList of MFCCs: {}'.format(data[MFCC]["value"]))
print()

MFCCs = np.asarray(data[MFCC]["value"])
n_seg = MFCCs.shape[0]
n_coef = MFCCs.shape[1]


def MFCCs_manager(MFCCs):
    m = np.mean(MFCCs, axis=0)
    sd = np.std(MFCCs, axis=0)
    d = np.ptp(MFCCs, axis=0)
    return m, sd, d


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

    print(m_norm)
    print(sd_norm)
    print(d_norm)


m, sd, d = MFCCs_manager(MFCCs)
input_normalizer(m, sd, d)
# print('\tMFCCs mean: {}'.format(m))
# print()
# print('\tMFCCs standard deviation: {}'.format(sd))
# print()
# print('\tMFCCs delta: {}'.format(d))
end = time.process_time()

print()
print('\tTime needed: {}'.format(end - start))

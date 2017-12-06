import json
from pprint import pprint


file_directory = "subtrainingset/TRAAABD128F429CF47.json"
json_data = open(file_directory).read()

GEN = "genre"
SEG_START = "segments_start"
MFCC = "segments_timbre"
SONG = "songs"

data = json.loads(json_data)

print('Track id: {}'.format(data[SONG]["value"][0][30]))
print('\tDuration: {} sec'.format(data[SONG]["value"][0][3]))
print('\tTempo: {} BPM'.format(data[SONG]["value"][0][27]))
print('\tKey: {}'.format(data[SONG]["value"][0][21]))
print('\tMusic genre: {}'.format(data[GEN]))
print('\tList of segment starts: {}'.format(data[SEG_START]["value"]))
print('\tList of MFCCs: {}'.format(data[MFCC]["value"]))



    # pprint(data["datasets"][item])
        #pprint(data["datasets"][item]["shape"]["dims"])
        #pprint(data["datasets"][item]["value"])
    # if data["datasets"][item]["alias"] == ['/analysis/segments_start']:
    #     pprint(data["datasets"][item]["shape"]["dims"])
    #     pprint(data["datasets"][item]["value"])
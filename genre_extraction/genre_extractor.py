import os
import json
from pprint import pprint

JSON_DIR = "../trainingset"
TRACK_FILE = "msd_genre_dataset.txt"

files = os.listdir(JSON_DIR)
track_ids = [file.split('.')[0] for file in files]

with open(TRACK_FILE, 'r', encoding='utf-8') as fl:
    data = fl.read()
    lines = data.split('\n')

    done = 0

    for track_id in track_ids:
        for line in lines:
            if track_id in line:

                if track_id == '':
                    continue

                genre = line.split(',')[0]
                file_name = JSON_DIR + '/' + track_id + '.json'

                to_store = {}

                with open(file_name, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)

                    to_store["genre"] = genre

                    for i in (data["datasets"].items()):
                        item = i[0]
                        if data["datasets"][item]["alias"] == ['/analysis/songs']:
                            to_store["songs"] = data["datasets"][item]
                        if data["datasets"][item]["alias"] == ['/analysis/segments_start']:
                            to_store["segments_start"] = data["datasets"][item]
                        if data["datasets"][item]["alias"] == ['/analysis/segments_timbre']:
                            to_store["segments_timbre"] = data["datasets"][item]

                    # pprint(to_store)
                    done += 1

                # os.remove(file_name)
                with open("../subtrainingset" + '/' + track_id + '.json', 'w', encoding='utf-8') as json_file:
                    json.dump(to_store, json_file)
                    print('Done {} of {}'.format(done, len(track_ids)))



    # for track_id in track_ids:
    # print(data.find(track_ids[0]))
import os
import json
JSON_DIR = "../trainingset"
TRACK_FILE = "msd_genre_dataset.txt"

files = os.listdir(JSON_DIR)
track_ids = [file.split('.')[0] for file in files]

with open(TRACK_FILE, 'r', encoding='utf-8') as fl:
    data = fl.read()
    lines = data.split('\n')
    for track_id in track_ids:
        for line in lines:
            if track_id in line:
                genre = line.split(',')[0]

                with open(JSON_DIR + '/' + track_id + '.json', 'rw', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    data["genre"] = genre
                    json.dump(data, json_file)


    # for track_id in track_ids:
    # print(data.find(track_ids[0]))
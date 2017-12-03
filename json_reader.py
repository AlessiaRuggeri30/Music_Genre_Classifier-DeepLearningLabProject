import json
from pprint import pprint
import numpy as np
import matplotlib as mpl


file_directory = "trainingset/TRAAAAW128F429D538.json"
json_data = open(file_directory).read()

data = json.loads(json_data)
to_store = {}

for i in (data["datasets"].items()):
    item = i[0]
    if data["datasets"][item]["alias"] == ['/analysis/songs']:
        to_store["/analysis/songs"] = data["datasets"][item]
    if data["datasets"][item]["alias"] == ['/analysis/segments_start']:
        to_store["/analysis/segments_start"] = data["datasets"][item]
    if data["datasets"][item]["alias"] == ['/analysis/segments_timbre']:
        to_store["/analysis/segments_timbre"] = data["datasets"][item]
pprint(to_store)
    # pprint(data["datasets"][item])
        #pprint(data["datasets"][item]["shape"]["dims"])
        #pprint(data["datasets"][item]["value"])
    # if data["datasets"][item]["alias"] == ['/analysis/segments_start']:
    #     pprint(data["datasets"][item]["shape"]["dims"])
    #     pprint(data["datasets"][item]["value"])



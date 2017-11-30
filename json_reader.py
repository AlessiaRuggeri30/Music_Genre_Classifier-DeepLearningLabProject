import json
from pprint import pprint
import numpy as np
import matplotlib as mpl


file_directory = "trainingset/TRAAAAW128F429D538.json"
json_data = open(file_directory).read()

data = json.loads(json_data)

for i in (data["datasets"].items()):
    item = i[0]
    if data["datasets"][item]["alias"] == ['/analysis/segments_timbre']:
        pprint(data["datasets"][item]["shape"]["dims"])
        pprint(data["datasets"][item]["value"])
    if data["datasets"][item]["alias"] == ['/analysis/segments_start']:
        pprint(data["datasets"][item]["shape"]["dims"])
        pprint(data["datasets"][item]["value"])
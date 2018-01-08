''' Script that runs h5tojson.py on all the files in all the folders
	that are contained in the root folder. In this way, al the files
	are converted from .h5 to .json format and saved in a new unique folder'''

import glob
import sys
import subprocess
import os

#Run on terminal:
#cd /Users/...(my user folder).../GitHub/Music_Genre_Classificator-DeepLearningLabProject/MillionSongSubset
#python3 main.py .

files = glob.glob(sys.argv[1] + '/**/*.h5', recursive=True)

for file in files:
    print('converting {}'.format(file))
    name = os.path.basename(file).split('.')[0]
    os.system('python3 h5tojson.py {} > ./trainingset/{}.json'.format(file, name))
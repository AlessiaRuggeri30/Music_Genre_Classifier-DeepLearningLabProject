import glob
import sys
import subprocess
import os

#Run on terminal:
#cd /Users/alessiaruggeri/GitHub/Music_Genre_Classificator-DeepLearningLabProject/MillionSongSubset
#python3 main.py .


files = glob.glob(sys.argv[1] + '/**/*.h5', recursive=True)

for file in files:
    print('converting {}'.format(file))
    name = os.path.basename(file).split('.')[0]
    os.system('python3 h5tojson.py {} > ./trainingset/{}.json'.format(file, name))
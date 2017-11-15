# Music_Genre_Classificator-DeepLearningLabProject

WORK IN PROGRESS

Deep Learning Lab project: Music Genre Classificator using audio features and neural networks

The aim is to create a neural network able to classify one or more songs by music genre, using some audio features extracted from the songs.
To do this, the song that has to be analized needs to get throught two main modules: the first one wants to extract the audio features needed for the genre classification from the file audio; the second one wants to use these features to classify the song with a neural network and to return the corresponding music genre as output.

Before having a working neural network model, we need a dataset go throught the training, the validation and the testing of the model. To train the model, it would be better to use a dataset of already extracted features and labelled songs by music genre.

This type of dataset is provided by the Million Song Dataset:
  https://labrosa.ee.columbia.edu/millionsong/

To have a little help for the features extraction, we could use some python library to manage and analyze the files audio.

PyAudioAnalysis could be a suitable library:
  https://github.com/tyiannak/pyAudioAnalysis

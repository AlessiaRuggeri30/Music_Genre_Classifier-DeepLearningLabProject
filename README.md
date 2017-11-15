# Music Genre Classificator - Deep Learning Lab Project

WORK IN PROGRESS

Group members: Alessia Ruggeri, Thomas Tiotto

Abstract: Music Genre Classificator using audio features and neural networks.

Introduction: The aim of the project is to create a neural network that is able to classify songs by music genre, using audio features extracted from them.

To do this, the song that has to be analysed needs to be fed through two main modules: the first one extracts the audio features needed for the genre classification from the file; the second uses these features to classify the song with a neural network and to return the corresponding music genre as its output.

Before having a working neural network model, we need a dataset go through the training, the validation and the testing of the model. To train the model, it would be better to use a dataset of already extracted features and labeled songs by music genre.

This type of dataset is provided by the Million Song Dataset:
    https://labrosa.ee.columbia.edu/millionsong/

To have a little help for the features extraction, we could use some python library to manage and analyse the audio files.

PyAudioAnalysis could be a suitable library:
    https://github.com/tyiannak/pyAudioAnalysis

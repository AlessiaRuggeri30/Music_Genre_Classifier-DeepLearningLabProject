## Music Genre Classifier - Deep Learning Lab Project

**WARNING**

This project has been implemented during my first months at USI, while I was still learning basic knowledge about deep learning; therefore the data analysis and the DL models could be better build with my current knowledge and, for this reason, I'm planning on working again on this project for my personal interest in the near future to improve the quality of the work and the results.

*Group members: Alessia Ruggeri, Thomas Tiotto*

*Abstract: Music Genre Classifier using audio features and neural networks*

The aim of the project is to implement a music genre classifier using neural networks. The classifier will receive as input a song’s audio signal and as output will return its music genre.

Standard feature extraction techniques relating to music information retrieval will be used. This information will be used to train a neural network, which will in turn be able to classify each input with its correct music genre. The architecture of the classifier will be as follows:

*AUDIO FILE  ->  FEATURE EXTRACTION  ->  NORMALIZATION   ->  FEED-FORWARD NEURAL NETWORK   ->  MUSIC GENRE*


### The training dataset

To train the model we will use a subset of the **Million Song Dataset** comprising 10,000 songs. This dataset provides a collection of features that have already been extracted from more than a million songs; these include genre and all the information needed to classify the songs. Here you can find the linlk to the official Million Song Dataset webpage: https://labrosa.ee.columbia.edu/millionsong/


### The feature extraction

To classify songs, we need to focus on the frequencies, the timbre and the dynamic of the audio signal. The Mel-Frequency Cepstrum (MFC), represented by **Mel-Frequency Cepstral Coefficients (MFCCs)**, is a feature that contains all these characteristics and that is sufficient to be able to classify songs based on their genre. MFCCs belong to the frequency domain: they are calculated by a Fourier Transform of the signal.

For each segment of a song, MFCCs are computed but only the first twelve coefficients are kept as the others are negligible. For each of the twelve vectors of coefficients, we compute mean, standard deviation and delta over the total number of segments in the song. In this way, we obtain 12x3=36 features for each song, that will be the inputs to our neural network.


### The neural network

The 36 features extracted will be the input of a feed-forward neural network. Initially, we thought about using a recurrent neural network based on LSTM cells and having the coefficients as inputs; we soon realised that a simpler feed-forward network would reduce computational complexity while giving the same results. The neural network will have 36 input neurons - one for each feature - and one output neuron for each music genre that we will decide to include.

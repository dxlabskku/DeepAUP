# DeepAUP: a deep neural network framework for abnormal underground heat transport pipelines

We propose a deep neural network framework for detecting abnormal underground heat transport pipelines (DeepAUP). DeepAUP could reduce amount of efforts for checking and maintaining underground heat transport pipelines. This model could be employed more efficient framework and real-time detection systems for urban areas in South Korea.

## Model
The overview of DeepAUP is presented in the below figure. First, we applied a set of convolutional layers to extract the features of MFCC. After extracting the MFCC feature values through convolutional layers, we attempted to learn sequential input information through the Bi-LSTM layers and Uni-LSTM layer. Finally, sequence-learned information was generated in the form of one-dimensional vectors by applying a fully connected layers.

![model](https://user-images.githubusercontent.com/96400041/173297481-5b039672-1b46-4c9a-a7d5-6310de7b53be.jpg)

## Data
We collected signal datasets of normal and abnormal pipelines and acoustic emission (AE) and acceleration (ACC) sensors were employed to measure the signals of heat transport pipelines. The sample of datasets are in ```data``` folder. Before training DeepAUP model, we have to do data preprocessing. First, we segmeneted each wave file into two-second periods. Then we extracted MFCC (Mel-Frequency Cepstral Coefficient) signal features. You could do data preprocessing with ```wavsplit.py``` and ```feature_extract.py``` files.   

## Experiment
If you wnat to train and test our model, you can execute the ```deepAUP.py``` file with MFCC feature data. The proposed DeepAUP model achieved over 99% accuracy and F1-score for all sensor datasets. With early-stopping method, DeepAUP can be rapidly performed, and because we could obtain both spatial and temporal features of pipeline signal data using the DeepAUP combined model that extracts spatial features through CNN and learns temporal features through Bi-LSTM. This approach is expected to improve the detection of pipeline abnormalities.

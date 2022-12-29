import os
import sys
import pickle
import numpy as np
from PIL import Image

# Architectural constants.
NUM_FRAMES = 512  # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.


# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 48000
STFT_WINDOW_LENGTH = 512
STFT_HOP_LENGTH = 256
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 150
MEL_MAX_HZ = 15000
EXAMPLE_WINDOW_SECONDS = 3.0  # Each example contains 96 10ms frames
EXAMPLE_HOP_SECONDS = 2.25  # with 25% overlap.


# Hyperparameters used in training.
INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.
ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.


## 

MODEL_TYPE = "buildNet"


##
RANDOM_SEED = 42


## indices of samples
df_SAMPLES = pickle.load(open("/content/drive/MyDrive/RP/DataSet/pkl/df_SAMPLES.pkl",'rb'))
sample_spec = pickle.load(open("/content/drive/MyDrive/RP/DataSet/pkl/org_spec.pkl",'rb'))
# aug_shift = pickle.load(open("/content/drive/MyDrive/RP/DataSet/pkl/aug_shift.pkl",'rb'))
com_aug_spec = pickle.load(open("/content/drive/MyDrive/RP/DataSet/pkl/new_spec.pkl",'rb'))

VAL_TEST_SAMPLES = df_SAMPLES.sample(frac=0.2, random_state=RANDOM_SEED, axis=0)
TRAIN_SAMPLES = df_SAMPLES.drop(index = VAL_TEST_SAMPLES.index)
VAL_SAMPLES = VAL_TEST_SAMPLES.sample(frac=0.5, random_state=RANDOM_SEED, axis=0)
TEST_SAMPLES = VAL_TEST_SAMPLES.drop(index = VAL_SAMPLES.index)

LABELS = sorted(TRAIN_SAMPLES.class_label.unique())

NB_CLASSES = len(LABELS)


# Definition of the input format
if MODEL_TYPE == 'resnet':
  RGB_INPUT = True # Does the chosen CNN takes RGB (3-channel) images as input?
else:
  RGB_INPUT = False

# What size should the images be resized to before they are fed to the CNN?
if MODEL_TYPE == 'buildNet':
  HEIGHT = 64
  WIDTH = 384
else:
  HEIGHT = 224
  WIDTH = 224

## Logging options
TENSORBOARD_LOGGING = True
CKPT_LOGGING = True
REDUCE_LR = True
EARLY_STOPPING = False

## 
BATCH_SIZE = 32
NB_EPOCHS = 50

##
OPTIMIZER = 'SGD'

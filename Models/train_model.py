import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard,ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow_addons.metrics import FBetaScore
from time import time
import random
from sklearn.utils import shuffle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

import params as params
from base_models import MODELS

################################

class DataGenerator(tf.keras.utils.Sequence):
    """
    Given their paths, it loads images and arranges them into batches of a proper format so that they can be directly be fed to the CNN.

    Output:
    Batches of images.
    """
    def __init__(self, spec_paths, labels):
        self.spec_paths = spec_paths # paths to images
        self.labels = labels # list of the corresponding labels (ground truth)
        self.n_classes = params.NB_CLASSES
        self.batch_size = params.BATCH_SIZE
        self.on_epoch_end()

    def __len__(self):
        # number of batches
        return int(np.floor(len(self.spec_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        spec_paths = [self.spec_paths[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        # Some CNNs only take RGB images as inputs
        if params.RGB_INPUT:
            X = np.empty((self.batch_size, int(params.HEIGHT), int(params.WIDTH), 3), dtype=np.int16)  # input size
        else:
            X = np.empty((self.batch_size, int(params.HEIGHT), int(params.WIDTH)), dtype=np.int16)  # input size
        y = np.empty((self.batch_size, self.n_classes), dtype=np.float32) #output size

        for i, (path, label) in enumerate(zip(spec_paths, labels)):
            img = Image.open(path)
            img = img.resize((params.WIDTH, params.HEIGHT))
            img = np.array(img)
            if params.RGB_INPUT:
                if len(list(img.shape)) == 2:
                    img = np.repeat(img[..., np.newaxis], 3, -1) # we fill the input image with 3 replicates of the spectrogram
            X[i,] = img
            y[i,] = to_categorical(label, num_classes=self.n_classes)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.spec_paths))

def get_train_val_data():
    train_samples = params.TRAIN_SAMPLES
    val_samples = params.VAL_SAMPLES

    train_filename = [item.split(".")[0] for item in train_samples['filename']]
    val_filename = [item.split(".")[0] for item in val_samples['filename']]

    train_specs = [path for path in params.sample_spec if path.split(os.sep)[-1].split("_")[0] in train_filename]
    # add data augmentation to training dataset
    train_specs += params.com_aug_spec
    val_specs = [path for path in params.sample_spec if path.split(os.sep)[-1].split("_")[0] in val_filename]
    
    # the lists are shuffled
    train_specs = shuffle(train_specs, random_state=params.RANDOM_SEED)
    val_specs = shuffle(val_specs, random_state=params.RANDOM_SEED)

    le = LabelEncoder()
    le.fit(params.LABELS)

    # In our case, labels are the names of the folders the images are located in
    train_labels = [path.split(os.sep)[-2] for path in train_specs]
    train_labels = le.transform(train_labels)

    val_labels = [path.split(os.sep)[-2] for path in val_specs]
    val_labels = le.transform(val_labels)
    
    # train_labels?train_specs
    assert len(train_specs) >= params.BATCH_SIZE, 'number of train samples must be >= batch_size'
    assert len(val_specs) >= params.BATCH_SIZE, 'number of val samples must be >= batch_size'

    #Training & validation batches
    tg = DataGenerator(train_specs, train_labels)
    vg = DataGenerator(val_specs, val_labels)

    return tg, vg


def get_model(model_type):

    base_model = MODELS[model_type] #Load base model

    for i, layer in enumerate(base_model.layers):
        layer.trainable = True
        print(i, layer.name)

    base_model.summary()

    return base_model


def get_logfiles(model_type, 
                checkpoint_log = params.CKPT_LOGGING, 
                tensorboard_log = params.TENSORBOARD_LOGGING,
                reducelr_log = params.REDUCE_LR,
                earlystopping_log = params.EARLY_STOPPING):
    logs = []
    
    if not os.path.isdir('/content/drive/MyDrive/RP/Li/Models/logs/'):
        os.mkdir('/content/drive/MyDrive/RP/Li/Models/logs/')
        
    if not os.path.isdir('/content/drive/MyDrive/RP/Li/Models/logs/csv/'):
        os.mkdir('/content/drive/MyDrive/RP/Li/Models/logs/csv/')

    csv_path = os.path.join('/content/drive/MyDrive/RP/Li/Models', 'logs', 'csv', '{}_history.csv'.format(model_type))    
    csv_logger = CSVLogger(csv_path, append=False)
    logs.append(csv_logger)

    if tensorboard_log:
        tensorboard = TensorBoard(log_dir="/content/drive/MyDrive/RP/Li/Models/logs/tb_logs/{}_{}".format(model_type, time()))
        # tensorboard --logdir = '/home/comp-land-eco/Documents/cnns/tb_logs'
        logs.append(tensorboard)

    if checkpoint_log:
        checkpoint = ModelCheckpoint('/content/drive/MyDrive/RP/Li/Models/logs/ckpt/{}_{}.h5'.format(model_type, time()), monitor='val_loss',
                             save_best_only=True, save_weights_only=False,
                             mode='auto', save_freq='epoch', verbose=1)
        logs.append(checkpoint)
  
    if reducelr_log:
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                    factor=0.5,
                                    patience=2,
                                    verbose=1, 
                                    mode='min')
        logs.append(reduce_lr)
    if earlystopping_log:
        early_stopping = EarlyStopping(monitor='val_loss', 
                                    verbose=1,
                                    patience=4)
        logs.append(early_stopping)

    return logs

def plot_history(history):
    his=pd.DataFrame(history.history)
    plt.subplots(1,2,figsize=(16,8))
    
    #loss:
    plt.subplot(1,2,1)
    plt.plot(range(len(his)),his['loss'],color='g',label='training')
    plt.plot(range(len(his)),his['val_loss'],color='r',label='validation')
    plt.legend()
    plt.title('Loss')
    
    #accuracy
    plt.subplot(1,2,2)
    plt.plot(range(len(his)),his['accuracy'],color='g',label='training_acc')
    plt.plot(range(len(his)),his['val_accuracy'],color='r',label='validation_acc')
    
#     #f1_score
#     plt.plot(range(len(his)),his['f1_score'],color='steelblue',label='training_f1')
#     plt.plot(range(len(his)),his['val_f1_score'],color='maroon',label='validation_f1')
    
    plt.legend()
    plt.title('accuracy')
    
    plt.show()       


def train():
    model_type = params.MODEL_TYPE
    base_model = get_model(model_type)


    # We add 2 dense layers on top: a 128-dimensional penultimate layer and a NB_CLASSES-dimensional final layer
    x = base_model.output
    x = Dense(128, activation='sigmoid')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(params.NB_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    f1_score=FBetaScore(num_classes=params.NB_CLASSES,average='macro',beta=1.0, name='f1_score')
    f05_score=FBetaScore(num_classes=params.NB_CLASSES,average='macro',beta=0.5, name='f0.5_score')
    # Compile the model and specify optimizer, loss and metric
    if params.OPTIMIZER=='SGD':
        model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.9),
                        metrics=["accuracy",f1_score,f05_score])
    else:
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.001),
                        metrics=["accuracy",f1_score,f05_score])

    tg, vg = get_train_val_data()

    his = model.fit(tg, validation_data = vg,
                epochs = params.NB_EPOCHS, 
                verbose=1,
                callbacks = get_logfiles(model_type))
    
    #plot_history(his)

    # Customize the name more if you want to train the same model several times!
    model.save("/content/drive/MyDrive/RP/Li/Models/trained_models/{}.h5".format(model_type))

if __name__ == '__main__':
    # avoid the cuDNN issue
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    train()

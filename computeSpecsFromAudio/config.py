# -*- coding: utf-8 -*-
"""
This file defines the parameters for spectrogram computation.
"""
import sys
#import pandas as pd



AUDIO_DIR = "/content/drive/MyDrive/RP/DataSet/recordings" # the folder containing audio files to be converted to spectrograms
SPEC_DIR = "/content/drive/MyDrive/RP/DataSet/specs" # the folder where you want the spectrogram images to be saved
SD_SPEC_DIR = "/content/drive/MyDrive/RP/DataSet/detected_specs"
LABEL_PATH = "/content/drive/MyDrive/RP/DataSet/labels.csv" # the csv file with species labels

AUG_SPEC_DIR = "/content/drive/MyDrive/RP/DataSet/aug_specs" 
AUG_SD_SPEC_DIR = "/content/drive/MyDrive/RP/DataSet/aug_detected_specs" 

'''
bird_data = pd.read_csv(LABEL_PATH)
print(bird_data)
print(bird_data.columns.values.tolist())
df2 = bird_data.groupby(['class_label'])['class_label'].count()
print(df2)
'''


## Parameters of Signal
##
SAMPLE_RATE = 48000 # Hz
DURATION = 15 # If DURATION == None, the files are not cut
OFFSET = 0.0 # Start time if cutting

##
CHUNK_SEC = 3
CHUNK_OVERLAP = 0.25

##
FMIN = 150
FMAX = 15000


## Parameters of Spectrogram
##
WIN_LEN = 512
WIN_OVERLAP = 0.50
##
N_MELS = 64
N_WIN = 512

##
RANDOM_SEED = 42

##
SIG_DETECTION = None#'median_clipping'

##
##
AUGMENTATION = True
SAMPLE_DICT = {"Saxicola torquatus_African Stonechat":214,
                "Perdix perdix_Grey Partridge":196,
                "Sylvia nisoria_Barred Warbler":97,
                "Lanius collurio_Red-backed Shrike":93,
                "Anthus pratensis_Meadow Pipit":66,
                "Alauda arvensis_Eurasian Skylark":58,
                "Emberiza hortulana_Ortolan Bunting":43,
                "Saxicola rubetra_Whinchat":0,
                "Linaria cannabina_Common Linnet":0,
                "Passer montanus_Eurasian Tree Sparrow":0,
                "Sylvia curruca_Lesser Whitethroat":0,
                "Emberiza calandra_Corn Bunting":0,
                "Motacilla flava_Western Yellow Wagtail":0,
                "Emberiza citrinella_Yellowhammer":0,
                "Sylvia communis_Common Whitethroat":0}
TRANSFORMS = {
    "train": [{"name": "PitchShift"},{"name": "TimeStretch"},{"name": "GaussianNoise"}]#,
    # "valid": [{"name": "Normalize"}],
    # "test": [{"name": "Normalize"}]
}

SPEC_DICT = {"Saxicola torquatus_African Stonechat":522,
                "Perdix perdix_Grey Partridge":483,
                "Sylvia nisoria_Barred Warbler":220,
                "Lanius collurio_Red-backed Shrike":425,
                "Anthus pratensis_Meadow Pipit":261,
                "Alauda arvensis_Eurasian Skylark":0,
                "Emberiza hortulana_Ortolan Bunting":170,
                "Saxicola rubetra_Whinchat":0,
                "Linaria cannabina_Common Linnet":154,
                "Passer montanus_Eurasian Tree Sparrow":0,
                "Sylvia curruca_Lesser Whitethroat":0,
                "Emberiza calandra_Corn Bunting":0,
                "Motacilla flava_Western Yellow Wagtail":0,
                "Emberiza citrinella_Yellowhammer":0,
                "Sylvia communis_Common Whitethroat":0}

SPEC_TRANSFORMS = {
    "spectrogram_transforms":{"train": [{"name": "TimeFreqMasking","params":{"time_drop_width":10,
                 "time_stripes_num":80, "freq_drop_width":2,"freq_stripes_num":5}},
                                        {"name": "TimeFreqShifting","params":{"shift_max":200,"roll":True}}]}
}

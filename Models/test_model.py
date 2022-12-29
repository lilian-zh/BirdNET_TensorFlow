import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
import params
import csv
from scipy.stats import rankdata
from time import time

from tensorflow_addons.metrics import FBetaScore



f1_score=FBetaScore(num_classes=params.NB_CLASSES,average='macro',beta=1.0, name='f1_score')
f05_score=FBetaScore(num_classes=params.NB_CLASSES,average='macro',beta=0.5, name='f0.5_score')

def load_model(path_to_saved_model):
    model = tf.keras.models.load_model(path_to_saved_model)
    return model


def get_array_from_img(path_to_image):
    img = Image.open(path_to_image)
    img = img.resize((params.WIDTH, params.HEIGHT))
    img_arr = np.array(img)
    if params.RGB_INPUT:
        if len(list(img_arr.shape)) == 2:
            img_arr = np.repeat(img_arr[..., np.newaxis], 3, -1)
    img_arr = np.expand_dims(img_arr, axis=0)  # Now shape is (1,params.WIDTH, params.HEIGHT,3)

    return img_arr


def get_prediction(model, input):
    preds = model.predict(input)
    return preds

def write_confusion_matrix(stats, save_cf_path):
    with open(os.path.join(save_cf_path), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['gt\pred']+params.LABELS)
        for species in params.LABELS:
            writer.writerow([species]+[i/sum(stats[species])*100 for i in stats[species]])

def write_mrr_score(mrr, Q, save_mrr_path):
    MRR = mrr/Q
    sample_size = Q

    if not os.path.isfile(save_mrr_path):
        with open(save_mrr_path, "w") as f:
          writer = csv.writer(f)
          writer.writerow(["Model type", "Time of test", "MRR score", "Sample size"])

    with open(save_mrr_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow([model, time(), mrr, Q])



if __name__ == '__main__':

    # avoid the cuDNN issue
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)


    model = params.MODEL_TYPE
    path_to_saved_model = '/content/drive/MyDrive/RP/Li/Models/trained_models/{}.h5'.format(model)
    model = load_model(path_to_saved_model)
    
    save_cf_path = '/content/drive/MyDrive/RP/Li/Models/test_stats/confusion_matrices/{}.csv'.format(model)
    save_mrr_path = '/content/drive/MyDrive/RP/Li/Models/test_stats/mrr_scores.csv'

    #initialization of the dictionary
    stats = {}
    mrr = 0
    Q = 0

    for species in params.LABELS:
        stats[species] = [0 for i in range(params.NB_CLASSES)]

    test_samples = params.TEST_SAMPLES

    for id_species, species in enumerate(params.LABELS):
        species_test_samples = test_samples.query('class_label == @species')
        if not species_test_samples.empty: 
            test_filename = [item.split(".")[0] for item in species_test_samples['filename']]
            test_specs = [path for path in params.sample_spec if path.split(os.sep)[-1].split("_")[0] in test_filename]
            for spec_path in test_specs:
                input_arr = get_array_from_img(spec_path)
                preds = get_prediction(model, input_arr)
                predicted_species_index = np.argmax(preds)

                # the stats dict is updated
                stats[species][predicted_species_index] += 1

                # update mrr and Q
                ranks_list = params.NB_CLASSES+1 - rankdata(list(preds)) # the highest probability score gets rank one
                rank_gt = ranks_list[id_species] #gives the rank of the ground truth species -> should be 1 if the CNN performed well
                mrr += 1/rank_gt
                Q += 1


    write_mrr_score(mrr, Q, save_mrr_path)
    write_confusion_matrix(stats, save_cf_path)
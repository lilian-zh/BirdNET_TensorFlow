import pandas as pd
import pickle

import config


#load meta-data file
bird_data = pd.read_csv(config.LABEL_PATH)
#selecting files used for training
# train_data = bird_data.query('split=="train"')

#Model Data Preparation:

bird_count = {}
for bird_species, count in zip(bird_data.class_label.unique(), bird_data.groupby("class_label")["class_label"].count().values):
    bird_count[bird_species] = count

'''
most_represented_birds = {bird for bird, count in bird_count.items() if count >= 175}
SAMPLES = bird_data.query('class_label in @most_represented_birds')
'''
SAMPLES = bird_data
LABELS = sorted(SAMPLES.class_label.unique())

print("Number of species", len(LABELS))
print("Number of samples", len(SAMPLES))
print(bird_count)


#Saving Labels
#with open('/content/drive/MyDrive/RP/DataSet/pkl/Labels.pkl', 'wb') as f:
#    pickle.dump(LABELS, f)


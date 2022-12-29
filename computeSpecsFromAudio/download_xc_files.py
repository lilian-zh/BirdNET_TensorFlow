import requests
import time
import csv
import datetime
import os
from pathlib import Path

start_time = datetime.datetime.now()

# read the defined species in csv file row by row, where 'gen' in column0 and 'sp' in column1
def read_species(species_path):
    species = csv.reader(open(species_path))
    # skip the heads
    next(species, None)  
    for line in species:
        yield line

# request for connet and obtain data in json 
def request_data(url):    
    r = requests.get(url,timeout=300)
    r_json = r.json()   
    return r_json


# extract json data and transform it to list of dir
def requestJson(species_path):
    recording_list = []
    numRecordings = 0
    species = read_species(species_path)
    for row in species:
        gen,sp = row[0],row[1]
        bird_dir = request_data('https://www.xeno-canto.org/api/2/recordings?query={}+{}+q:A'.format(gen,sp))
        numRecordings += int(bird_dir['numRecordings'])
        for item in bird_dir['recordings']:
            recording_list.append(item)
    print(numRecordings)
    return recording_list

# download recordings and metadata
def download(species_path,data_dir,recording_path,split):   
    recording_list = requestJson(species_path)

    metadata_file = Path(data_dir + "/labels.csv")
    if metadata_file.is_file():
        open_mode = "a+"
    else:
        open_mode = "w"
        
    with open(metadata_file, mode=open_mode,newline='') as f:
        w = csv.writer(f)
        # solve the problem to automatically write the header
        headers=["class_label","filename","split"]
        if open_mode == "w":
            w.writerow(headers)
        for item in recording_list:
            file_name = "XC"+item['id']+"."+item['file-name'].split(".")[-1]
            w.writerow([item['gen']+" "+item['sp']+"_"+item['en'],file_name,split])
        
            r_audio = requests.get(item['file'], timeout=300)
            #download recordings
            with open(r"{}/{}".format(recording_path,file_name), 'wb') as audio:
                audio.write(r_audio.content)
            time.sleep(1)

data_dir = "/projects/p094/p_kpml/LiZhang"

recording_path = os.path.join(data_dir + "/recordings")

species_path = os.path.join(data_dir + "/species.csv")

download(species_path,data_dir,recording_path,split="train")

end_time = datetime.datetime.now()
print("start_time:{},end_time:{},gap:{}".format(start_time,end_time,(end_time-start_time).seconds))


  
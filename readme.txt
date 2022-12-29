This project is an implementation of BirdNET via tensorflow. Some changes in preprocessing and test part were made.


######################################################
Run the script to download audio files from xeno-canto:

python download_xc_files.py



Select the parameters you want in the computeSpecsFromAudio/config.py file and run getSpecs with the command:

python getSpecs.py



If data augmentation is required, run the command:

python aug.py


#######################################################
Select the parameters you want in the Models/params.py file and run script to train the model with the command:

python train_model.py


For prediction:

python test_model.py



Citation
https://github.com/kahst/BirdNET



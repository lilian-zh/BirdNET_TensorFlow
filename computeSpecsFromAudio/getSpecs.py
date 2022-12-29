import cv2
import numpy as np
import pandas as pd
import librosa
import os
import traceback
import scipy
import warnings
warnings.filterwarnings('ignore')
import pickle
from PIL import Image
from tqdm import tqdm
import scipy.ndimage as ndimage

import config
import cleanLabels as cl



def openAudioFile(audio_path):
    
    # Open file with librosa (uses ffmpeg or libav)
    sig, rate = librosa.load(audio_path, sr=config.SAMPLE_RATE, offset=config.OFFSET, duration=config.DURATION, res_type='kaiser_fast')

    return sig, rate


def splitSignal(sig):

    chunk_len = int(config.CHUNK_SEC * config.SAMPLE_RATE)
    step_len = int((config.CHUNK_SEC * (1-config.CHUNK_OVERLAP)) * config.SAMPLE_RATE)

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), step_len): 
        split = sig[i:i + chunk_len]

        # End of signal?
        if len(split) < chunk_len:
            break
        
        sig_splits.append(split)

    return sig_splits


def applyBandpassFilter(sig, order=4):

    wn = np.array([config.FMIN, config.FMAX]) / (config.SAMPLE_RATE / 2.0)
    filter_sos = scipy.signal.butter(order, wn, btype='bandpass', output='sos')
    filtered = scipy.signal.sosfiltfilt(filter_sos, sig)

    return filtered


def compute_spec(sig,bandpass=False):

    # Bandpass filter?
    if bandpass:
        sig = applyBandpassFilter(sig)

    # Compute overlap
    hop_len = int(config.WIN_LEN*(1-config.WIN_OVERLAP))
    # Compute spectrogram
    mel_spec = librosa.feature.melspectrogram(y=sig, 
                                            sr=config.SAMPLE_RATE, 
                                            n_fft=config.WIN_LEN, 
                                            hop_length=hop_len, 
                                            n_mels=config.N_MELS, 
                                            fmin=config.FMIN, 
                                            fmax=config.FMAX)

    # Log-Mel Spectrogram 
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Flip spectrum vertically (only for better visialization, low freq. at bottom)
    # mel_spec = mel_spec[::-1, ...]

    # Trim to desired shape if too large
    mel_spec = mel_spec[:config.N_MELS, :config.N_WIN]
 
    # Normalize values between 0 and 1
    mel_spec -= mel_spec.min()
    if not mel_spec.max() == 0:
        mel_spec /= mel_spec.max()
    else:
        mel_spec = np.clip(mel_spec, 0, 1)

    return mel_spec

#Remove single spots from an image
def filter_isolated_cells(array, struct):

    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array, structure=struct)
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    
    return filtered_array


#Decide if given spectrum shows bird sounds or noise only
def median_clip(spec, mark_path, threshold=10):

    #working copy
    img = spec.copy()

    #STEP 1: Median blur
    img = cv2.medianBlur(img,5)

    #STEP 2: Median threshold
    col_median = np.median(img, axis=0, keepdims=True)
    row_median = np.median(img, axis=1, keepdims=True)

    img[img < row_median * 1] = 0
    img[img < col_median * 1.5] = 0
    img[img > 0] = 1

    #STEP 3: Remove singles
    img = filter_isolated_cells(img, struct=np.ones((3,3)))

    #STEP 4: Morph Closing
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,5), np.float32))

    #STEP 5: Frequency crop
    #img = img[128:-16, :]

    #STEP 6: Count columns and rows with signal
    #(Note: We only use rows with signal as threshold, but columns might come in handy in other scenarios)

    #column has signal?
    col_max = np.max(img, axis=0)
    col_max = ndimage.morphology.binary_dilation(col_max, iterations=2).astype(col_max.dtype)
    cthresh = col_max.sum()

    #row has signal?
    row_max = np.max(img, axis=1)
    row_max = ndimage.morphology.binary_dilation(row_max, iterations=2).astype(row_max.dtype)
    rthresh = row_max.sum()

    #final threshold
    thresh = rthresh

    #DBUGB: show?
    #print thresh
    #cv2.imshow('BIRD?', img)
    #cv2.waitKey(-1)

    #formatted = (img * 255 / np.max(img)).astype('uint8')
    #img = Image.fromarray(formatted)
    #img.show()
    #img.save(mark_path)
    

    #STEP 7: Apply threshold (Default = 16)
    bird = True
    if thresh < threshold:
        bird = False
    #print(mark_path,thresh,bird)

    return bird, thresh


def specsFromFile(audio_path, label, spec_dir):

    # Open file
    sig, rate = openAudioFile(audio_path)

    # Split signal in consecutive chunks with overlap
    sig_splits = splitSignal(sig)

    #extract mel-spectrograms for each chunks
    s_cnt = 0
    saved_samples = []
    
    # try:
    for chunk in sig_splits:
        mel_spec = compute_spec(chunk)

        if config.SIG_DETECTION == 'median_clipping':
            mark_path = os.path.join(spec_dir,'specs_test',audio_path.rsplit(os.sep, 1)[-1].rsplit('.', 1)[0] + 
                                '_' + str(s_cnt)  + '.png')
            spec = np.array(mel_spec, dtype="float32")
            #does spec contain bird sounds?
            # rejected specs will be copied to "noise" folder
            isbird, thresh = median_clip(spec, mark_path)
            if isbird:
                pass
                # save_dir = os.path.join(spec_dir, label)
            else:
                save_dir = os.path.join(spec_dir, 'non-event')
                if not os.path.exists(save_dir):
                  os.makedirs(save_dir)

                save_path = os.path.join(save_dir, audio_path.rsplit(os.sep, 1)[-1].rsplit('.', 1)[0] + 
                                        '_' + str(s_cnt)  + '.png')

                im = Image.fromarray(mel_spec * 255.0).convert("L")
                im.save(save_path)
                saved_samples.append(save_path)
                s_cnt += 1

        # else:
        #     save_dir = os.path.join(spec_dir, label)

        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

        # save_path = os.path.join(save_dir, audio_path.rsplit(os.sep, 1)[-1].rsplit('.', 1)[0] + 
        #                         '_' + str(s_cnt)  + '.png')

        # im = Image.fromarray(mel_spec * 255.0).convert("L")
        # im.save(save_path)
        # saved_samples.append(save_path)
        # s_cnt += 1

    # except:
    #     print(s_cnt, "ERROR")
    #     traceback.print_exc()
    #     pass

    # sig_samples = [path for path in saved_samples if "noise" not in path]
    return saved_samples#sig_samples



if __name__ == "__main__":

    sample_spec = []
    #bad_audios=[]
    df_SAMPLES = pd.DataFrame(columns=list(cl.SAMPLES))

    print('INITIAL NUMBER OF AUDIO FILES:', len(cl.SAMPLES))

    with tqdm(total=len(cl.SAMPLES)) as pbar:
        for idx, row in cl.SAMPLES.iterrows():
            pbar.update(1)
            
            #if row.class_label in cl.SAMPLES:
            audio_path = os.path.join(config.AUDIO_DIR, row.filename)
            # in case there are bad audios which can not be opened
            #try:
            if config.SIG_DETECTION == 'median_clipping':
                spec_dir = config.SD_SPEC_DIR
            else:
                spec_dir = config.SPEC_DIR

            sample_spec += specsFromFile(audio_path, row.class_label, spec_dir)
            df_SAMPLES = df_SAMPLES.append(row)
            #except Exception:
            #    bad_audios.append(row.filename) 
            #    continue

    # # fn_SAMPLES = shuffle(df_SAMPLES, random_state = config.RANDOM_SEED)
    # print('FINAL NUMBER OF AUDIO FILES:', len(df_SAMPLES))

    # # Saving df_SAMPLES
    # with open('/content/drive/MyDrive/RP/DataSet/pkl/df_SAMPLES.pkl', 'wb') as f:
    #     pickle.dump(df_SAMPLES, f)
           
    # fn_SPECS = shuffle(sample_spec, random_state=config.RANDOM_SEED)
    print('SUCCESSFULLY EXTRACTED {} SPECTROGRAMS'.format(len(sample_spec)))

    # Saving sample_spec
    with open('/content/drive/MyDrive/RP/DataSet/pkl/noise_spec.pkl', 'wb') as f:
        pickle.dump(sample_spec, f)

    






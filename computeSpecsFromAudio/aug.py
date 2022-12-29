import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from PIL import Image

from random import choice
import os
import random
import pickle

import config




def get_transforms(phase: str):
    transforms = config.TRANSFORMS
    if transforms is None:
        return None
    else:
        if transforms[phase] is None:
            return None
        trns_list = []
        for trns_conf in transforms[phase]:
            trns_name = trns_conf["name"]
            trns_params = {} if trns_conf.get("params") is None else \
                trns_conf["params"]
            if globals().get(trns_name) is not None:
                trns_cls = globals()[trns_name]
                trns_list.append(trns_cls(**trns_params))

        if len(trns_list) > 0:
            return Compose(trns_list)
        else:
            return None


def get_waveform_transforms(phase: str):
    return get_transforms(phase)


def get_spectrogram_transforms(config: dict, phase: str):
    transforms = config.get('spectrogram_transforms')
    if transforms is None:
        return None
    else:
        if transforms[phase] is None:
            return None
        trns_list = []
        for trns_conf in transforms[phase]:
            trns_name = trns_conf["name"]
            trns_params = {} if trns_conf.get("params") is None else \
                trns_conf["params"]
            if hasattr(A, trns_name):
                trns_cls = A.__getattribute__(trns_name)
                trns_list.append(trns_cls(**trns_params))
            else:
                trns_cls = globals().get(trns_name)
                if trns_cls is not None:
                    trns_list.append(trns_cls(**trns_params))

        if len(trns_list) > 0:
            return A.Compose(trns_list, p=1.0)
            # return A.Compose(trns_list,p=0.5)
        else:
            return None
            

class Normalize:
    def __call__(self, y: np.ndarray):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)


class NewNormalize:
    def __call__(self, y: np.ndarray):
        y_mm = y - y.mean()
        return y_mm / y_mm.abs().max()


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20, sr=48000):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented



class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_range=5, sr=48000):
        super().__init__(always_apply, p)
        self.max_range = max_range
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        n_steps = np.random.randint(-self.max_range, self.max_range)
        augmented = librosa.effects.pitch_shift(y, self.sr, n_steps)
        return augmented


class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1, sr=48000):
        super().__init__(always_apply, p)
        self.max_rate = max_rate
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented

class OneOf:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        n_trns = len(self.transforms)
        trns_idx = np.random.choice(n_trns)
        trns = self.transforms[trns_idx]
        y = trns(y)
        return y


def drop_stripes(image: np.ndarray, dim: int, drop_width: int, stripes_num: int):
    total_width = image.shape[dim]
    lowest_value = image.min()
    for _ in range(stripes_num):
        distance = np.random.randint(low=0, high=drop_width, size=(1,))[0]
        begin = np.random.randint(
            low=0, high=total_width - distance, size=(1,))[0]

        if dim == 0:
            image[begin:begin + distance] = lowest_value
        elif dim == 1:
            image[:, begin + distance] = lowest_value
        elif dim == 2:
            image[:, :, begin + distance] = lowest_value
    return image


class TimeFreqMasking(ImageOnlyTransform):
    def __init__(self,
                 time_drop_width: int,
                 time_stripes_num: int,
                 freq_drop_width: int,
                 freq_stripes_num: int,
                 always_apply=False,
                 p=0.5):
        super().__init__(always_apply, p)
        self.time_drop_width = time_drop_width
        self.time_stripes_num = time_stripes_num
        self.freq_drop_width = freq_drop_width
        self.freq_stripes_num = freq_stripes_num

    def apply(self, img, **params):
        img_ = img.copy()
        if img.ndim == 2:
            img_ = drop_stripes(
                img_, dim=0, drop_width=self.freq_drop_width, stripes_num=self.freq_stripes_num)
            img_ = drop_stripes(
                img_, dim=1, drop_width=self.time_drop_width, stripes_num=self.time_stripes_num)
        return img_



def image_shifting(img, shift_max=200, roll=True):
    # assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'
    img = img.copy()
    dir_list = ['right', 'left', 'down', 'up']
    direction = choice(dir_list)
    if direction == ['right','left']:
        shift = np.random.randint(low=1, high=shift_max, size=(1,))[0]
    else:
        shift = np.random.randint(low=1, high=int(shift_max/10), size=(1,))[0]

    if direction == 'right':
        right_slice = img[:, -shift:].copy()
        img[:, shift:] = img[:, :-shift]
        if roll:
            img[:,:shift] = np.fliplr(right_slice)
    if direction == 'left':
        left_slice = img[:, :shift].copy()
        img[:, :-shift] = img[:, shift:]
        if roll:
            img[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = img[-shift:, :].copy()
        img[shift:, :] = img[:-shift,:]
        if roll:
            img[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = img[:shift, :].copy()
        img[:-shift, :] = img[shift:, :]
        if roll:
            img[-shift:,:] = upper_slice

    return img

class TimeFreqShifting(ImageOnlyTransform):
    def __init__(self,
                 shift_max=200, 
                 roll=True,
                 always_apply=False,
                 p=0.5):
        super().__init__(always_apply, p)
        self.shift_max = shift_max
        self.roll = roll

    def apply(self, img, **params):
        img_ = img.copy()
        img_ = image_shifting(img_, shift_max=self.shift_max, roll=self.roll)

        return img_




 
# sig, rate = librosa.load(r"C:\Users\Lili\Downloads\XC562345.mp3", sr=config.SAMPLE_RATE, offset=config.OFFSET, duration=config.DURATION, res_type='kaiser_fast')
# print("org sig:",sig)
# get_trns = get_waveform_transforms("train")
# new_sig = get_trns(y=sig)
# print("new_sig:",new_sig)

# img_path = r"C:\Users\Lili\Downloads\XC575136_5.png"
# img = Image.open(img_path)
# img_arr = np.array(img)
# get_trns = get_spectrogram_transforms(config=config.SPEC_TRANSFORMS, phase="train")
# new_arr = get_trns(image=img_arr)['image']
# im = Image.fromarray(new_arr).convert("L")
# im.save(r"C:\Users\Lili\Downloads\XC575136_5_aug.png")



pre_dir = "/content/drive/MyDrive/RP/DataSet/detected_specs"
org_spec = []
new_spec = []

for (key,value) in config.SPEC_DICT.items():
    org_dir = os.path.join(pre_dir, key)
    files = os.listdir(org_dir)   
    for spec in files:
        org_path = os.path.join(org_dir, spec)
        org_spec.append(org_path)
    if value>0:
      choose_files = np.random.choice(files, value, replace=True)
      counter = 0
      for item in choose_files:
          choose_path = os.path.join(org_dir, item)

          img = Image.open(choose_path)
          img_arr = np.array(img)
          get_trns = get_spectrogram_transforms(config=config.SPEC_TRANSFORMS, phase="train")
          new_arr = get_trns(image=img_arr)['image']
          im = Image.fromarray(new_arr).convert("L")
          save_dir = os.path.join("/content/drive/MyDrive/RP/DataSet/com_aug_specs",key)
          if not os.path.exists(save_dir):
              os.makedirs(save_dir)
          save_path = os.path.join(save_dir,item.split(".")[0]+'_aug_'+str(counter)+'.png')
          im.save(save_path)
          new_spec.append(save_path)

with open('/content/drive/MyDrive/RP/DataSet/pkl/org_spec.pkl', 'wb') as f:
    pickle.dump(org_spec, f)

with open('/content/drive/MyDrive/RP/DataSet/pkl/new_spec.pkl', 'wb') as f:
    pickle.dump(new_spec, f)



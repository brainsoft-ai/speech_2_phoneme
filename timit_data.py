import numpy as np
from python_speech_features import mfcc, fbank, delta
from sklearn.preprocessing import StandardScaler
import scipy.io.wavfile as wav
import subprocess
import os, time, pickle

import torch
import joblib

from glob import glob
from tqdm import tqdm
from utils import load_dj_spectrogram as read_djs


phn_61 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh',
          'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 
          'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl',
          'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
phn_61 = ['<sos>', '<eos>'] + phn_61

mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix',
               'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n',
               'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#',
               'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#',
               'q': 'h#', 'epi': 'h#', 'pau': 'h#'}

phn_39 = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 
             'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', 
             'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw',
             'v', 'w', 'y', 'z', 'zh']
phn_39 = ['<sos>', '<eos>'] + phn_39

development_set = ['FAKS0', 'MMDB1', 'MBDG0', 'FEDW0', 'MTDT0', 'FSEM0', 'MDVC0', 'MRJM4', 'MJSW0', 'MTEB0',
                  'FDAC1', 'MMDM2', 'MBWM0', 'MGJF0', 'MTHC0', 'MBNS0', 'MERS0', 'FCAL1', 'MREB0', 'MJFC0',
                  'FJEM0', 'MPDF0', 'MCSH0', 'MGLB0', 'MWJG0', 'MMJR0', 'FMAH0', 'MMWH0', 'FGJD0', 'MRJR0',
                  'MGWT0', 'FCMH0', 'FADG0', 'MRTK0', 'FNMR0', 'MDLS0', 'FDRW0', 'FJSJ0', 'FJMG0', 'FMML0',
                  'MJAR0', 'FKMS0', 'FDMS0', 'MTAA0', 'FREW0', 'MDLF0', 'MRCS0', 'MAJC0', 'MROA0', 'MRWS1']

core_test_set = ['MDAB0', 'MWBT0', 'FELC0', 'MTAS1', 'MWEW0', 'FPAS0', 'MJMP0', 'MLNT0', 'FPKT0',
             'MLLL0', 'MTLS0', 'FJLM0', 'MBPM0', 'MKLT0', 'FNLP0', 'MCMJ0', 'MJDH0', 'FMGD0',
            'MGRT0', 'MNJM0', 'FDHC0', 'MJLN0', 'MPAM0', 'FMLD0']

TIMIT_DIR = '/brain/data/TIMIT/original_TIMIT' # root directory for timit, it would be joined with timit/train or timit/test

feat_type = 'fbank'
feats_dim = 39 if feat_type=='mfcc' else 123
labels_sos_id = len(phn_61) + 1
labels_eos_id = len(phn_61) 
num_classes = len(phn_61) + 1

keep_prob = 0.5
n_encoder_layer = 3
beam_width = 10
batch_size = 32

def create_cahce_data(set_name='train',
                      feats_type='mfcc',
                      remake=False,
                     ):
    start = time.time()

    data_fname = os.path.join('cache', f'{set_name}_{feats_type}_data.pkl')
    scaler_fname = os.path.join('cache', f'{feats_type}_scaler.pkl')
    
    if set_name == 'train':
        filter_fn = lambda file, _: file.startswith('SA')
        root_dir = os.path.join(TIMIT_DIR, 'TRAIN')
    if set_name == 'dev':
        filter_fn = lambda file, path: file.startswith('SA') or os.path.split(path)[1] not in development_set
        root_dir = os.path.join(TIMIT_DIR, 'TEST')
    if set_name == 'test':
        filter_fn = lambda file, path: file.startswith('SA') or os.path.split(path)[1] not in core_test_set
        root_dir = os.path.join(TIMIT_DIR, 'TEST')

    if not remake and os.path.exists(data_fname):
        feats_list, phoneme_list = joblib.load(data_fname)
        print(f'{set_name} loaded: {len(phoneme_list)} utterances - {time.time()-start:.0f}s')
        return feats_list, phoneme_list 

    feats_list, phoneme_list = [], []

    for path, dirs, files in os.walk(root_dir):
        for file in files:
            if filter_fn(file, path):
                continue

            if file.endswith('WAV'):
                fullFileName = os.path.join(path, file)
                fnameNoSuffix = os.path.splitext(fullFileName)[0]
                fNameTmp = fnameNoSuffix + '_tmp.wav'
                # convert nist file format to wav with command line program 'sox'
                subprocess.call(f"sox {fullFileName} {fNameTmp}", shell=True)
                rate, sig = wav.read(fNameTmp)
                os.remove(fNameTmp)

                if feats_type == 'mfcc':
                    mfcc_feat = mfcc(sig, rate)
                    mfcc_feat_delta = delta(mfcc_feat, 2)
                    mfcc_feat_delta_delta = delta(mfcc_feat_delta, 2)
                    feats = np.concatenate((mfcc_feat, mfcc_feat_delta, mfcc_feat_delta_delta), axis=1)
                elif feats_type == 'fbank':
                    filters, energy = fbank(sig, rate, nfilt=40)
                    log_filters, log_energy = np.log(filters), np.log(energy)
                    logfbank_feat = np.concatenate((log_filters, log_energy.reshape(-1,1)), axis=1)
                    logfbank_feat_delta = delta(logfbank_feat, 2)
                    logfbank_feat_delta_delta = delta(logfbank_feat_delta, 2)
                    feats = np.concatenate((logfbank_feat, logfbank_feat_delta, logfbank_feat_delta_delta), axis=1)
                feats_list.append(feats)

                # .phn
                phoneme = []
                with open(fnameNoSuffix + '.PHN', 'r') as f:
                    for line in f.read().splitlines():
                        phn = line.split(' ')[2]
                        p_index = phn_61.index(phn)
                        phoneme.append(p_index)

                phoneme_list.append(phoneme)
                
    if set_name == 'train':
        concat_features = np.concatenate(feats_list, axis=0)
        means = concat_features.mean(axis=0)
        stds = concat_features.std(axis=0)
        joblib.dump([means, stds], scaler_fname)

    if not os.path.exists(scaler_fname):
        raise Exception('scaler.pkl not exist, call with [train_set=True]')
    else:
        means, stds = joblib.load(scaler_fname)

    for idx, feats in enumerate(feats_list):
        feats_list[idx] = (feats - means) / stds

    joblib.dump([feats_list, phoneme_list], data_fname)
    print(f'{set_name} created: {len(phoneme_list)} utterances - {time.time()-start:.0f}s')


    return feats_list, phoneme_list 
    

class timit(torch.utils.data.Dataset):
    def __init__(self,
                feats_type='mfcc',
                set_name='train',
                preproc=['add_eos'], # 
                # preproc=['normalization', 'add_eos'], # 
                aug=[],
                remake=False
        ):
        self.set_name = set_name

        # todo: remove hard code
        featlen_max, phonemelen_max = 778, 75

        self.x, self.y = create_cahce_data(set_name, feats_type, remake=remake)

        # preproc constant
        self.add_eos = True if 'add_eos' in preproc else False

        # augmentation constant
        self.repeat_aug = True if 'repeat' in aug else False

        c_repeat = 5 if self.repeat_aug else 1 # to repeat sentence augmentation
        c_eos = 1 if self.add_eos else 0  # add end token

        self.featlen_max = c_repeat * featlen_max + c_eos
        self.phonemelen_max = c_repeat * phonemelen_max + c_eos

    def __getitem__(self, index):
        each_data = np.zeros([self.featlen_max, 123], dtype=np.float32)
        each_target = np.zeros([self.phonemelen_max], dtype=np.long)

        n_x = len(self.x[index])
        n_y = len(self.y[index])

        num_repeat = 1
        if self.repeat_aug:
            low = 1
            high = 6
            num_repeat = np.random.randint(low, high)
            
        for idx in range(num_repeat):
            each_data[idx*n_x:(idx+1)*n_x] = self.x[index]
            each_target[idx*n_y:(idx+1)*n_y] = self.y[index]
            
        if self.add_eos:
            each_data[num_repeat*n_x] = 0
            each_target[num_repeat*n_y] = 1 # <eos>

        return each_data, each_target

    def __len__(self):
        return len(self.x)

if __name__ == "__main__":

    feats_type = 'fbank'
    remake = False

    # aug = ['repeat']
    aug = []
    iterator = timit(feats_type, set_name='train', aug=aug, remake=remake)
    iterator = timit(feats_type, set_name='dev', aug=aug, remake=remake)
    iterator = timit(feats_type, set_name='test', aug=aug, remake=remake)

    for x,y in iterator:
        pass

    for idx, each in enumerate(phn_61):
        print(idx, each, end=' ')
    print('done')

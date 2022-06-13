from ast import parse
import os 
import glob
from shutil import ExecError
from librosa import feature
import numpy as np 
import librosa
import librosa.display
import matplotlib.pyplot as plt 
import argparse
import sys 
import librosa.display

import argparse

parser = argparse.ArgumentParser(description="PyTorch")
parser.add_argument('--loc', type=str, required=True, help='location of data - [ground / buried ]')
parser.add_argument('--data_path', type=str, default='/data/minyoung/pipe/2022/kict/wav2sec', help='target data directory')
parser.add_argument('--target_root', type=str, default='/data/minyoung/pipe/2022/kict/features', help='target root directory')
parser.add_argument('--feature_type', type=str, default='STFT', help='feature type')
parser.add_argument('--n_fft', type=int, default=4096, help='n_fft')
parser.add_argument('--win_length', type=int, default=4096, help='win_length')
parser.add_argument('--hop_length', type=int, default=512, help='hop_length')
parser.add_argument('--n_mels', type=int, default=256, help='n_mels')
parser.add_argument('--n_mfcc', type=int, default=30, help='n_mfcc')

opt = parser.parse_args()


loc = opt.loc
data_path = opt.data_path
target_root = opt.target_root
feature_type = opt.feature_type
n_fft = opt.n_fft
win_length = opt.win_length
hop_length = opt.hop_length
n_mels = opt.n_mels
n_mfcc = opt.n_mfcc

target_root = os.path.join(target_root, loc)
if not os.path.exists(target_root):
    os.mkdir(target_root)

output_dir_name = "MFCC_nfft%d_winlength%d_hoplength%d_nmfcc%d"%(n_fft,win_length,hop_length,n_mfcc)
output_dir_name = os.path.join(target_root,output_dir_name)
output_normal_dir_name = os.path.join(output_dir_name,"normal")
output_leak_dir_name = os.path.join(output_dir_name,"leak")

if not os.path.exists(output_dir_name):
    os.mkdir(output_dir_name)
if not os.path.exists(output_normal_dir_name):
    os.mkdir(output_normal_dir_name)
if not os.path.exists(output_leak_dir_name):
    os.mkdir(output_leak_dir_name)
print(output_dir_name)
print(output_normal_dir_name)
print(output_leak_dir_name)
# if input("continue?") == "q": exit()

normal_paths = glob.glob(os.path.join(data_path, 'normal/{}/*.wav'.format(loc)))
leak_paths = glob.glob(os.path.join(data_path, 'leak/{}/*.wav'.format(loc)))

i = 0
print("generating MFCC normal: total - {}".format(len(normal_paths)))
for normal_path in normal_paths:
    print("%d/%d"%(i+1,len(normal_paths)),end='\r')
    i += 1
    y, sr = librosa.load(normal_path)
    # wav_vis_image_name = "WAV_"+os.path.splitext(os.path.basename(normal_path))[0]+".png"
    # wav_vis_image_path = os.path.join(output_normal_dir_name,wav_vis_image_name)
    # librosa.display.waveshow(y,sr,alpha=0.4)
    # plt.savefig(wav_vis_image_path,dpi=300)
    # plt.clf()
    D = np.abs(librosa.stft(y, n_fft = n_fft, win_length = win_length, hop_length = hop_length))
    mfcc = librosa.feature.mfcc(S = librosa.power_to_db(D), sr = sr, n_mfcc = n_mfcc)
    mfcc_db = librosa.amplitude_to_db(mfcc, ref=0.00002)
    output_file_name = os.path.splitext(os.path.basename(normal_path))[0]+".npy"
    output_file_name = os.path.join(output_normal_dir_name,output_file_name)
    np.save(output_file_name,mfcc_db)

i = 0
print("generating MFCC leak: total - {}".format(len(leak_paths)))
for leak_path in leak_paths:
    print("%d/%d"%(i+1,len(leak_paths)),end='\r')
    i += 1
    y, sr = librosa.load(leak_path)
    # wav_vis_image_name = "WAV_"+os.path.splitext(os.path.basename(leak_path))[0]+".png"
    # wav_vis_image_path = os.path.join(output_leak_dir_name,wav_vis_image_name)
    # librosa.display.waveshow(y,sr,alpha=0.4)
    # plt.savefig(wav_vis_image_path,dpi=300)
    # plt.clf()
    D = np.abs(librosa.stft(y, n_fft = n_fft, win_length = win_length, hop_length = hop_length))
    mfcc = librosa.feature.mfcc(S = librosa.power_to_db(D), sr = sr, n_mfcc = n_mfcc)
    mfcc_db = librosa.amplitude_to_db(mfcc, ref=0.00002)
    output_file_name = os.path.splitext(os.path.basename(leak_path))[0]+".npy"
    output_file_name = os.path.join(output_leak_dir_name,output_file_name)
    np.save(output_file_name,mfcc_db)
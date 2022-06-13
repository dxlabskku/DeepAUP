import os 
import numpy as np
import soundfile as sf
import librosa 
import glob
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="PyTorch")
parser.add_argument('--type', type=str, required=True, help='which type of data - [normal / leak]')
parser.add_argument('--loc', type=str, required=True, help='location of data - [ground / buried ]')
parser.add_argument('--data_path', type=str, default='/data/minyoung/pipe/2022/kict/', help='target data directory')
parser.add_argument('--target_root', type=str, default='/data/minyoung/pipe/2022/kict/wav2sec', help='target root directory')
opt = parser.parse_args()

type = opt.type
loc = opt.loc
data_path = opt.data_path
target_root = opt.target_root

data_list = os.listdir(data_path)
data_list = [d for d in data_list if d[0] in ['1','2','3','4','5','6','7','8','9'] and d[-1] != 'p']
assert(type == 'leak' or type == 'normal')
if type == 'leak':
    file_list = [d for d in data_list if '누수' in d]
else:
    file_list = [d for d in data_list if '정상' in d]

target_data_path = os.path.join(target_root, type)
print(target_data_path)
if not os.path.exists(target_data_path):
    os.mkdir(target_data_path)

assert(loc == 'ground' or loc == 'buried')
if loc == 'ground':
    file_path_list = [os.path.join(data_path, n) for n in file_list if '지상' in n]
else:
    file_path_list = [os.path.join(data_path, n) for n in file_list if '매립' in n]

targetdir = os.path.join(target_data_path, loc)
if not os.path.exists(targetdir):
    os.mkdir(targetdir)
    

for l in file_path_list:
    print(l)
    file_path = l
    file_data_list = os.listdir(file_path)

    for n in file_data_list:
        print('  ', n)
        n_path = os.path.join(file_path, n)
        n_list = os.listdir(n_path)

        for data in tqdm(n_list):
            print('    ', data)
            inputdir = os.path.join(n_path, data)
            data_list = os.listdir(inputdir)

            for wav in data_list:
                print('      ', wav)
                wav_file_path = os.path.join(inputdir, wav)
                y,sr = librosa.load(wav_file_path) 
                idx = 0 # 자르기 시작하는 wav 파일의 시작 초 
                dist = 2 # 자르고자 하는 wav 파일의 길이 
                while True:
                    if (idx+dist) * sr > y.shape[0]: # wav 파일의 맨 끝인 경우 
                        cur_sample = y[int(idx*sr):] # 남은 모든 신호를 추출 
                        break
                    else:
                        print('{}/{}'.format(idx, y.shape[0]))
                        cur_sample = y[int(idx*sr):int((idx+dist)*sr)]
                    sf.write(os.path.join(targetdir,"%d.wav"%(len(os.listdir(targetdir))+1)),cur_sample,sr,subtype='PCM_16') # 파일 저장 
                    idx += 2
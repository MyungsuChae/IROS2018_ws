import os, re, random, sys
from os.path import join, splitext, basename
from glob import glob
import numpy as np
from process import *
random.seed(123)

import torch
import torch.utils.data as data

import soundfile as sf
from skimage import io, transform

class IEMOCAP(data.Dataset):
    def __init__(self, which_set, datatype='multi', datapath='/hdd1/home/thkim/data/IEMOCAP_full_release/preprocessed', shuffle=True, subset=''):
        self.__dict__.update(locals())

        self.__audio_mean__ = -4.181317587822257e-05
        self.__audio_var__ = 0.003423544945266016
        self.wholedatalist = []
        total_duration = 0
        exclude_duration = 0
        with open(join(datapath, 'timing.txt'), 'r') as f:
            for line in f:
                _, fname, _, duration =  line.strip().split('\t')
                duration = float(duration)
                if duration > 8:
                    exclude_duration += duration
                    continue
                total_duration += duration
                self.wholedatalist.append(fname)

        self.wholedatalist = [join(datapath, xx) for xx in self.wholedatalist]
          
        if 'impro' in subset:
            self.wholedatalist = [xx for xx in self.wholedatalist if 'impro' in xx]

        if 'session1' in subset:
            self.wholedatalist = [xx for xx in self.wholedatalist if 'Ses01' in xx]

        if 'toy' in subset:
            self.wholedatalist = self.wholedatalist[:200]

        if shuffle:
            random.shuffle(self.wholedatalist)
        self.datalist = {
            'train': self.wholedatalist[:int(len(self.wholedatalist)*0.6)],
            'valid': self.wholedatalist[int(len(self.wholedatalist)*0.6):int(len(self.wholedatalist)*0.8)],
            'test': self.wholedatalist[int(len(self.wholedatalist)*0.8):]
        }[which_set]

        if which_set == 'train':
            print('[*] Total duration {:.0f}s, and {:.0f}s are excluded'.format(total_duration, exclude_duration))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        fname = self.datalist[index]
        rawname =  fname.split('/')[-1]
        assert rawname.startswith('Ses')

        x_a = self.audioread(join(fname, rawname + '.wav'))
        x_v = self.videoread(sorted(glob(join(fname, '*.jpg'))))
        y = self.avdread(join(fname, rawname + '.avd'))
        emo = self.emoread(join(fname, rawname + '.cat'))
        gen = self.genread(join(fname, rawname + '.gen'))
        return x_a, x_v, y, emo, gen 
       
    def audioread(self, fname):
        gen = fname.split('_')[-1][0]
        x_a, fs = sf.read(fname)
        x_a = x_a[:,0] if gen=='F' else x_a[:,1]
        x_a = torch.tensor(x_a).view(1,-1)
        x_a = (x_a - self.__audio_mean__)/self.__audio_var__
        return x_a

    def videoread(self, flist):
        x_v = np.zeros((len(flist), 3, 96, 96))
        for i, fname in enumerate(flist):
            temp_x = np.transpose(io.imread(fname), (2,0,1))
            x_v[i] = temp_x
        x_v = torch.tensor(x_v).reshape(len(flist)*3, 96, 96)
        return x_v
        
    def avdread(self, fname):
        with open(fname, 'r') as f:
            line = f.readline().split(',')
        avd = list(map(float, line))
        avd = torch.tensor(avd)- 3
        return avd
    
    def emoread(self, fname):
        with open(fname, 'r') as f:
            emo = int(f.readline().strip())
        emo = torch.tensor(int(emo))
        return emo

    def genread(self, fname):
        with open(fname, 'r') as f:
            gen = int(f.readline().strip())
        gen = torch.tensor(gen)
        return gen

def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    x_a, x_v, y, emo, gen= zip(*data)
    n_batch = len(x_a)
    
    aperv = 640
    video_lengths = [int(len(xx)/3) for xx in x_v]
    audio_lengths = [int(len(xx)/3 * aperv) for xx in x_v]

    max_v_length = max(video_lengths) * 3
    max_a_length = max(audio_lengths)

    video = torch.zeros(n_batch, max_v_length, 96, 96)
    audio = torch.zeros(n_batch, 1, max_a_length)
    label = torch.stack(y, 0) 
    emo = torch.stack(emo, 0)
    gen = torch.stack(gen, 0)

    for i, (a, v) in enumerate(zip(x_a, x_v)):
        v_end = video_lengths[i] * 3
        a_end = audio_lengths[i]

        video[i, :v_end, :, :] = v
        if len(a[0]) > a_end:
            audio[i, :, :a_end] = a[0][:a_end]
        else:
            audio[i, :, :len(a[0])] = a 
        
    return  video, audio, torch.LongTensor(video_lengths), emo, gen


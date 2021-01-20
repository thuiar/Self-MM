import os
import random
import pickle
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

__all__ = ['MMDataLoader']

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosi,
            'sims': self.__init_sims,
        }
        DATA_MAP[args.datasetName](args)

    def __init_mosi(self, args):
        with open(args.dataPath, 'rb') as f:
            data = pickle.load(f)
        if 'use_bert' in args and args.use_bert:
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id'][:,0].tolist()
        if 'aligned' not in self.args or not self.args.aligned:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']
        self.audio[self.audio == -np.inf] = 0
        self.labels = {
            'M': data[self.mode]['labels'].astype(np.float32)
        }
        print(f"{self.mode} samples: {self.labels['M'].shape}")
        if 'need_normalize' in args and args.need_normalize:
            self.train_visual_max = np.max(np.max(np.abs(data['train']['vision']), axis=0), axis=0)
            self.train_visual_max[self.train_visual_max==0] = 1
            self.__normalize()
    
    def __init_mosei(self, args):
        return self.__init_mosi(args)

    def __init_sims(self, args):
        with open(args.dataPath, 'rb') as f:
            data = pickle.load(f)
        if 'use_bert' in args and args.use_bert:
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['feature_T'].astype(np.float32)
        self.vision = data[self.mode]['feature_V'].astype(np.float32)
        self.audio = data[self.mode]['feature_A'].astype(np.float32)
        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['ids']
        if 'aligned' not in self.args or not self.args.aligned:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']
        self.labels = {
            'M': data[self.mode]['label_M'],
            'T': data[self.mode]['label_T'], 
            'A': data[self.mode]['label_A'], 
            'V': data[self.mode]['label_V']
        }
        print(f"{self.mode} samples: {self.labels['M'].shape}")
        if 'need_normalize' in args and args.need_normalize:
            self.train_visual_max = np.max(np.max(np.abs(data['train']['feature_V']), axis=0), axis=0)
            self.train_visual_max[self.train_visual_max==0] = 1
            self.__normalize()

    def __truncated(self):
        # NOTE: Here for dataset we manually cut the input into specific length.
        def Truncated(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if((instance[index] == padding).all()):
                        if(index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+20])
                            break
                    else:                        
                        truncated_feature.append(instance[index:index+20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature
                       
        text_length, audio_length, video_length = self.args.input_lens
        self.vision = Truncated(self.vision, video_length)
        self.text = Truncated(self.text, text_length)
        self.audio = Truncated(self.audio, audio_length)

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.vision = np.mean(self.vision, axis=0, keepdims=True)
        self.audio = np.mean(self.audio, axis=0, keepdims=True)

        # remove possible NaN values
        self.vision[self.vision != self.vision] = 0
        self.audio[self.audio != self.audio] = 0

        self.vision = np.transpose(self.vision, (1, 0, 2))
        self.audio = np.transpose(self.audio, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if 'use_bert' in self.args and self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.rawText[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            # 'id': self.ids[index].decode('UTF-8'),
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        if 'aligned' not in self.args or not self.args.aligned:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        return sample

def MMDataLoader(args):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'input_lens' in args:
        args.input_lens = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader
import os
import argparse

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

__all__ = ['Config']

class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used inadition to `obj['foo']`
    ref: https://blog.csdn.net/a200822146085/article/details/88430450
    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __str__(self):
        return "<" + self.__class__.__name__ + dict.__repr__(self) + ">"

class Config():
    def __init__(self, args):
        # parameters for data
        # global parameters for running
        self.globalArgs = args
        # hyper parameters for models
        self.HYPER_MODEL_MAP = {
            'self_mm': self.__SELF_MM
        }
        # hyper parameters for datasets
        self.HYPER_DATASET_MAP = self.__datasetCommonParams()
    
    def __datasetCommonParams(self):
        assert self.globalArgs.datasetName in ['mosi', 'mosei', 'sims']

        tmp = "aligned" if self.globalArgs.aligned else "unaligned"
        if self.globalArgs.datasetName in ['mosi', 'mosei']:
            text_len = 50
        elif self.globalArgs.datasetName in ['sims']:
            text_len = 39
            
        dataPath = os.path.join(self.globalArgs.data_dir, self.globalArgs.datasetName, \
                                tmp + '_' + str(text_len) + '.pkl')
        tmp = {
            'mosi':{
                'dataPath': dataPath,
                'input_lens': (50, 50, 50) if self.globalArgs.aligned else (50, 500, 375),
                # (text, audio, video)
                'feature_dims': (768, 5, 20),
                'train_samples': 1284,
                'language': 'en',
                'KeyEval': 'Loss'
            },
            'mosei':{
                'dataPath': dataPath,
                'input_lens': (50, 50, 50) if self.globalArgs.aligned else (50, 500, 375),
                # (text, audio, video)
                'feature_dims': (768, 74, 35),
                'train_samples': 16326,
                'language': 'en',
                'KeyEval': 'Loss'
            },
            'sims':{
                'dataPath': dataPath,
                # (batch_size, input_lens, feature_dim)
                'input_lens': (39, 400, 55), # (text, audio, video)
                'feature_dims': (768, 33, 709), # (text, audio, video)
                'train_samples': 1368,
                'language': 'cn',
                'KeyEval': 'Loss',
            },
        }
        return tmp

    def __SELF_MM(self):
        tmp = {
            'commonParas':{
                'need_align': False,
                'need_normalize': False,
                'use_bert': True,
                'use_finetune': True,
                'early_stop': 12,
                'task_type': 'regression', # regression / classification
                'tasks': 'M'
            },
            # dataset
            'datasetParas':{
                'mosi':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'update_epochs': 4,
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 1e-3,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.001,
                    'weight_decay_other': 0.001,
                    'num_classes': 1,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':64,
                    'post_audio_dim': 16,
                    'post_video_dim': 32,
                    'post_fusion_dropout': 0.1,
                    'post_text_dropout': 0.0,
                    'post_audio_dropout': 0.1,
                    'post_video_dropout': 0.1,
                    # res
                    'H': 3.0
                },
                'mosei':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'update_epochs': 4,
                    'batch_size': 32,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 0.005,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 1e-3,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.0,
                    'weight_decay_video': 0.0,
                    'weight_decay_other': 0.01,
                    'num_classes': 1,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 32,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 128,
                    'post_text_dim':32,
                    'post_audio_dim': 16,
                    'post_video_dim': 16,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.0,
                    'post_video_dropout': 0.1,
                    # res
                    'H': 3.0
                },
                'sims':{
                    # the batch_size of each epoch is update_epochs * batch_size
                    'update_epochs': 4,
                    'batch_size': 16,
                    'learning_rate_bert': 5e-5,
                    'learning_rate_audio': 0.005,
                    'learning_rate_video': 0.005,
                    'learning_rate_other': 0.001,
                    'weight_decay_bert': 0.001,
                    'weight_decay_audio': 0.01,
                    'weight_decay_video': 0.0,
                    'weight_decay_other': 0.001,
                    'num_classes': 1,
                    # feature subNets
                    'a_lstm_hidden_size': 32,
                    'v_lstm_hidden_size': 64,
                    'a_lstm_layers': 1,
                    'v_lstm_layers': 1,
                    'text_out': 768, 
                    'audio_out': 16,
                    'video_out': 32, 
                    'a_lstm_dropout': 0.0,
                    'v_lstm_dropout': 0.0,
                    't_bert_dropout':0.1,
                    # post feature
                    'post_fusion_dim': 64,
                    'post_text_dim':64,
                    'post_audio_dim': 16,
                    'post_video_dim': 16,
                    'post_fusion_dropout': 0.0,
                    'post_text_dropout': 0.1,
                    'post_audio_dropout': 0.0,
                    'post_video_dropout': 0.1,
                    # res
                    'H': 1.0
                },
            },
        }
        return tmp

    def get_config(self):
        # normalize
        model_name = str.lower(self.globalArgs.modelName)
        dataset_name = str.lower(self.globalArgs.datasetName)
        # integrate all parameters
        res =  Storage(dict(vars(self.globalArgs),
                            **self.HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            **self.HYPER_MODEL_MAP[model_name]()['commonParas'],
                            **self.HYPER_DATASET_MAP[dataset_name]))
        return res
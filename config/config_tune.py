import os
import random
import argparse

from utils.functions import Storage

class ConfigTune():
    def __init__(self, args):
        # global parameters for running
        self.globalArgs = args
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'self_mm': self.__SELF_MM
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()
        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['debugParas'],
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = '/home/sharing/disk3/dataset/multimodal-sentiment-dataset/StandardDatasets'
        tmp = {
            'mosi':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 5, 20),
                    'train_samples': 1284,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss' 
                }
            },
            'mosei':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/aligned_50.pkl'),
                    'seq_lens': (50, 50, 50),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'MOSEI/Processed/unaligned_50.pkl'),
                    'seq_lens': (50, 500, 375),
                    # (text, audio, video)
                    'feature_dims': (768, 74, 35),
                    'train_samples': 16326,
                    'num_classes': 3,
                    'language': 'en',
                    'KeyEval': 'Loss'
                }
            },
            'sims':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/Processed/features/unaligned_39.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (39, 400, 55), # (text, audio, video)
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                }
            }
        }
        return tmp

    def __SELF_MM(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
                'save_labels': False,
                'early_stop': 8,
                'update_epochs': 4,
            },
            'debugParas':{
                'd_paras': ['batch_size', 'learning_rate_bert','learning_rate_audio', 'learning_rate_video', \
                            'learning_rate_other', 'weight_decay_bert', 'weight_decay_other', 
                            'weight_decay_audio', 'weight_decay_video',\
                            'a_lstm_hidden_size', 'v_lstm_hidden_size', 'text_out', 'audio_out', 'video_out',\
                            'a_lstm_dropout', 'v_lstm_dropout', 't_bert_dropout', 'post_fusion_dim', 'post_text_dim', 'post_audio_dim', \
                            'post_video_dim', 'post_fusion_dropout', 'post_text_dropout', 'post_audio_dropout', 'post_video_dropout', 'H'],
                'batch_size': random.choice([16, 32]),
                'learning_rate_bert': random.choice([5e-5]),
                'learning_rate_audio': random.choice([1e-4, 1e-3, 5e-3]),
                'learning_rate_video': random.choice([1e-4, 1e-3, 5e-3]),
                'learning_rate_other': random.choice([1e-4, 1e-3]),
                'weight_decay_bert': random.choice([0.001, 0.01]),
                'weight_decay_audio': random.choice([0.0, 0.001, 0.01]),
                'weight_decay_video': random.choice([0.0, 0.001, 0.01]),
                'weight_decay_other': random.choice([0.001, 0.01]),
                # feature subNets
                'a_lstm_hidden_size': random.choice([16, 32]),
                'v_lstm_hidden_size': random.choice([32, 64]),
                'a_lstm_layers': 1,
                'v_lstm_layers': 1,
                'text_out': 768,
                'audio_out': random.choice([16]),
                'video_out': random.choice([32]), 
                'a_lstm_dropout': random.choice([0.0]),
                'v_lstm_dropout': random.choice([0.0]),
                't_bert_dropout':random.choice([0.1]),
                # post feature
                'post_fusion_dim': random.choice([64, 128]),
                'post_text_dim':random.choice([32, 64]),
                'post_audio_dim': random.choice([16, 32]),
                'post_video_dim': random.choice([16, 32]),
                'post_fusion_dropout': random.choice([0.1, 0.0]),
                'post_text_dropout': random.choice([0.1, 0.0]),
                'post_audio_dropout': random.choice([0.1, 0.0]),
                'post_video_dropout': random.choice([0.1, 0.0]),
            }
        }
        return tmp

    def get_config(self):
        return self.args

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--debug_mode', type=bool, default=False,
                            help='adjust parameters ?')
        parser.add_argument('--modelName', type=str, default='ef_lstm',
                            help='support mult/tfn/lmf/mfn/ef_lstm/lf_dnn/graph_mfn/mtfn/mlmf/mlf_dnn')
        parser.add_argument('--datasetName', type=str, default='sims',
                            help='support mosi/sims')
        parser.add_argument('--tasks', type=str, default='M',
                            help='M/T/A/V/MTAV/...')
        parser.add_argument('--num_workers', type=int, default=8,
                            help='num workers of loading data')
        parser.add_argument('--model_save_path', type=str, default='results/model_saves',
                            help='path to save model.')
        parser.add_argument('--res_save_path', type=str, default='results/result_saves',
                            help='path to save results.')
        parser.add_argument('--data_dir', type=str, default='/home/sharing/disk3/dataset/multimodal-sentiment-dataset',
                            help='path to data directory')
        parser.add_argument('--gpu_ids', type=list, default=[2],
                            help='indicates the gpus will be used.')
        return parser.parse_args()
        
    args = parse_args()
    config = ConfigTune(args)
    args = config.get_config()
    print(args)
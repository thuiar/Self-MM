import os
import time
import random
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from config.config_run import Config
from config.config_debug import ConfigDebug
from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args):
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    # device
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    print("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device
    # data
    dataloader = MMDataLoader(args)
    model = AMIO(args).to(device)

    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
        return answer

    print(f'The model has {count_parameters(model)} trainable parameters')
    # exit()
    # using multiple gpus
    if using_cuda and len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model,
                                      device_ids=args.gpu_ids,
                                      output_device=args.gpu_ids[0])
    # start running
    # do train
    atio = ATIO().getTrain(args)
    # do train
    atio.do_train(model, dataloader)
    # load pretrained model
    pretrained_path = os.path.join(args.model_save_path,\
                                        f'{args.modelName}-{args.datasetName}-{args.tasks}.pth')
    assert os.path.exists(pretrained_path)
    model.load_state_dict(torch.load(pretrained_path))
    model.to(device)
    # do test
    if args.debug_mode:
        # using valid dataset to debug hyper parameters
        results = atio.do_test(model, dataloader['valid'], mode="VALID")
    else:
        results = atio.do_test(model, dataloader['test'], mode="TEST")
    """
    eval_results = {
            "has0_acc_2":  acc2,
            "has0_F1_score": f_score,
            "non0_acc_2":  non_zeros_acc2,
            "non0_F1_score": non_zeros_f1_score,
            "Mult_acc_5": mult_a5,
            "Mult_acc_7": mult_a7,
            "MAE": mae,
            "Correlation Coefficient": corr,
        }
    """
    return results

def run_debug(seeds, debug_times=50):
    has_debuged = [] # save used paras
    args = parse_args()
    config = ConfigDebug(args)
    save_file_path = os.path.join(args.res_save_path, \
                                args.datasetName + '-' + args.modelName + '-' + args.tasks + '-debug.csv')
    if not os.path.exists(os.path.dirname(save_file_path)):
        os.makedirs(os.path.dirname(save_file_path))
    
    errorReturn = 3 # try 3 times when encountering exception
    for i in range(debug_times):
        # cancel random seed
        setup_seed(int(time.time()))
        args = config.get_config()
        # print debugging params
        print("#"*40 + '%s-(%d/%d)' %(args.modelName, i+1, debug_times) + '#'*40)
        for k,v in args.items():
            if k in args.d_paras:
                print(k, ':', v)
        print("#"*90)
        print('Start running %s...' %(args.modelName))
        # restore existed paras
        if i == 0 and os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
            for i in range(len(df)):
                has_debuged.append([df.loc[i,k] for k in args.d_paras])
        # check paras
        cur_paras = [args[v] for v in args.d_paras]
        if cur_paras in has_debuged:
            print('These paras have been used!')
            time.sleep(3)
            continue
        try:
            has_debuged.append(cur_paras)
            results = []
            for j, seed in enumerate(seeds):
                args.cur_time = j + 1
                setup_seed(seed)
                results.append(run(args)['M'])
            errorReturn = 3 # reset
        except Exception as e:
            if "out of memory" in str(e):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            print(e)
            if errorReturn:
                time.sleep(3)
                errorReturn -= 1
                continue
            else:
                return
        # save results to csv
        print('Start saving results...')
        if os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
        else:
            df = pd.DataFrame(columns = [k for k in args.d_paras] + [k for k in results[0].keys()])
        # stat results
        tmp = [args[c] for c in args.d_paras]
        for col in results[0].keys():
            values = [r[col] for r in results]
            tmp.append(round(sum(values) * 100 / len(values), 2))

        df.loc[len(df)] = tmp
        df.to_csv(save_file_path, index=None)
        print('Results are saved to %s...' %(save_file_path))

def run_normal(seeds):
    model_results = []
    # run results
    for i, seed in enumerate(seeds):
        args = parse_args()
        args.cur_time = i+1
        # load config
        config = Config(args)
        args = config.get_config()
        setup_seed(seed)
        args['seed'] = seed
        print('Start running %s...' %(args.modelName))
        print(args)
        # runnning
        test_results = run(args)
        # restore results
        model_results.append(test_results)
    # save results
    criterions = list(model_results[0][args.tasks[0]].keys())
    df = pd.DataFrame(columns=["Model"] + criterions)
    # for m in args.tasks:
    for m in ['M']:
        res = [args.modelName+'-'+m]
        for c in criterions:
            values = [r[m][c] for r in model_results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            res.append((mean, std))
        df.loc[len(df)] = res
    save_path = os.path.join(args.res_save_path, \
                    args.datasetName + '-' + args.modelName + '-' + args.tasks + '.csv')
    if not os.path.exists(args.res_save_path):
        os.makedirs(args.res_save_path)
    df.to_csv(save_path, index=None)
    print('Results are saved to %s...' %(save_path))
            
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug_mode', type=bool, default=False,
                        help='adjust parameters ?')
    parser.add_argument('--modelName', type=str, default='self_mm',
                        help='support bert_mult/bert_misa/bert_misa/self_mm')
    parser.add_argument('--aligned', type=bool, default=False,
                        help='need aligned data ?')
    parser.add_argument('--datasetName', type=str, default='sims',
                        help='support mosi/mosei/sims')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers of loading data')
    parser.add_argument('--model_save_path', type=str, default='results/model_saves',
                        help='path to save model.')
    parser.add_argument('--res_save_path', type=str, default='results/result_saves',
                        help='path to save results.')
    parser.add_argument('--data_dir', type=str, default='/home/sharing/disk3/dataset/multimodal-sentiment-dataset/ALL',
                        help='path to data directory')
    parser.add_argument('--gpu_ids', type=list, default=[3],
                        help='indicates the gpus will be used.')
    return parser.parse_args()

if __name__ == '__main__':
    seeds = [1111, 1112, 1113, 1114, 1115]
    if parse_args().debug_mode:
        run_debug(seeds, debug_times=50)
    else:
        run_normal(seeds)
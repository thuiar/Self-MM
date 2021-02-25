![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

## SELF-MM
> Pytorch implementation for codes in [Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis (AAAI2021)](https://arxiv.org/abs/2102.04830). Please see our another repo [MMSA](https://github.com/thuiar/MMSA) for more details, which is a scalable framework for MSA.

### Model

![model](assets/MainModel.png)

### Usage

1. Download datasets and preprocessing
- MOSI and MOSEI
> download from [CMU-MultimodalSDK](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/)

- SIMS
> download from [Baidu Yun Disk](https://pan.baidu.com/s/1CmLdhYSVnNFAyA0DkR6tdA)[code: `ozo2`] or [Google Drive](https://drive.google.com/file/d/1z6snOkOoy100F33lzmHHB_DUGJ47DaQo/view?usp=sharing)

Then, preprocess data and save as a pickle file with the following structure using `data/DataPre.py`
```python
{
    "train": {
        "raw_text": [],
        "audio": [],
        "vision": [],
        "id": [], # [video_id$_$clip_id, ..., ...]
        "text": [],
        "text_bert": [],
        "audio_lengths": [],
        "vision_lengths": [],
        "annotations": [],
        "classification_labels": [], # Negative(< 0), Neutral(0), Positive(> 0)
        "regression_labels": []
    },
    "valid": {***}, # same as the "train" 
    "test": {***}, # same as the "train"
}
```

2. Download [Bert-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) from [Google-Bert](https://github.com/google-research/bert).  
Then, convert Tensorflow into pytorch using [transformers-cli](https://huggingface.co/transformers/converting_tensorflow_models.html) and move them into `pretrained_model`  

2. Clone this repo and install requirements.
```
git clone https://github.com/thuiar/Self-MM
cd Self-MM
conda create --name self_mm python=3.7
source activate self_mm
pip install -r requirements.txt
```

4. Make some changes
Modify the `config/config_tune.py` and `config/config_regression.py` to update dataset pathes.

3. Run codes
```
python run.py --modelName self_mm --datasetName mosi
```

### Results

> Detailed results are shown in [MMSA](https://github.com/thuiar/MMSA) > [results/result-stat.md](https://github.com/thuiar/MMSA/blob/master/results/result-stat.md). 

### Paper
---
Please cite our paper if you find our work useful for your research:
```
@inproceedings{yu2021le,
  title={Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis},
  author={Yu, Wenmeng and Xu, Hua and Ziqi, Yuan and Jiele, Wu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```

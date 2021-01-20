![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
# SELF-MM
> Pytorch implementation for codes in [Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis (AAAI2021)]()


### Usage
> This repo is similar to our previous work, [MMSA](https://github.com/thuiar/MMSA).

---

- Clone this repo and install requirements.
```
git clone https://github.com/thuiar/Self-MM
cd Self-MM
pip install -r requirements.txt
```

- Run codes
```
python run.py --modelName self_mm --datasetName mosi
```

### Results
- MOSI

| Model     | MAE   | Corr  | Acc-2 | F1-Score |
| :---:     | :---: | :---: | :---: | :---:    |
| BERT-MULT |  |  | | |
| BERT-MISA |  |  | | |
| BERT-MAG  |  |  | | |
| SELF-MM   |  |  | | |

- MOSEI

| Model     | MAE | Corr | Acc-2 | F1-Score |
| :---:     | :---: | :---: | :---: | :---: |
| BERT-MULT |  |  | |
| BERT-MISA |  |  | |
| BERT-MAG  |  |  | |
| SELF-MM   |  |  | |

- SIMS

| Model     | MAE | Corr | Acc-2 | F1-Score |
| :---:     | :---: | :---: | :---: | :---: |
| BERT-MULT |  |  | |
| BERT-MISA |  |  | |
| BERT-MAG  |  |  | |
| SELF-MM   |  |  | |


### Paper
---
Please cite our paper if you find our work useful for your research:
```
@inproceedings{yu2021le,
  title={Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis},
  author={Yu, Wenmeng and Xu, Hua and Ziqi, Yuan and Jiele, Wu},
  year={2021}
}
```
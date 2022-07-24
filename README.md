# AcTune
This is the code repo for our paper `[AcTune: Uncertainty-Based Active Self-Training for Active Fine-Tuning of Pretrained Language Models](https://aclanthology.org/2022.naacl-main.102/)' (In Proceedings of NAACL 2022 Main Conference, Oral Presentation).

# Requirements
```
python 3.7
transformers==4.2.0
pytorch==1.6.0
tqdm
scikit-learn
faiss-cpu==1.6.4

```
# Datasets
## Main Experiments
We use the following four datasets for the main experiments.
|   Dataset   | Task  | Number of Classes | Number of Train/Test |
|---------------- | -------------- |-------------- | -------------- |
| [SST-2](https://nlp.stanford.edu/sentiment/)       |     Sentiment           |     2   |  60.6k/1.8k  |
| [AG News](https://huggingface.co/datasets/ag_news) |    News Topic       |      4      |  119k/7.6k   |
| [Pubmed-RCT](https://github.com/Franck-Dernoncourt/pubmed-rct)  |   Medical Abstract   |     5        |     180k/30.1k    |
| [DBPedia](https://huggingface.co/datasets/dbpedia_14)     |     Ontology Topic      |      14      |     280k/70k      |

The processed data can be found at [this link](https://drive.google.com/drive/folders/1Yhsf1Gji-kCxPfFtNnNW8isSbXqKZcKh?usp=sharing). The folder to put these datasets will be discribed in the following parts.

## Weak Supervision Experiments
Most of the dataset are from the [WRENCH](https://github.com/JieyuZ2/wrench) benchmark. Please checkout their repo for dataset details.

# Training
Please use the commands in `commands` folder for experiments.
Take AG News dataset as an example, `run_agnews_finetune.sh` is used for running the experiment of standard active learning approaches, and `run_agnews_finetune.sh` is used for running active self-training experiments as unlabeled data is also used during fine-tuning.

Here, we suppose there is a folder for storing datasets as in `../datasets/`, and a folder for logging the experiment results as in `../exp`. 

# Hyperparameter Tuning
The key hyperparameter for our approach includes `pool`, `pool_min`, `gamma`, `gamma_min`, `self_training_weight` `sample_per_group` and `n_centroids`.
- `pool` stands for the number of samples selected for self-training *on average* for each round. For example, if 
- `pool_min` stands for the initial number of samples selected for self-training. For example, if `pool_min=3000` and `pool=4000` and there are 10 rounds in total, it means that in the first round, it selects 3000 samples, and in the later rounds, the number of low-uncertainty samples used for self-training will increase linearly. Finally, the total number of unlabeled data used for self-training equals to `4000*10=40000`.
- `gamma` stands for the final weight of momentum-based memory bank.
- `gamma_min` stands for the initial weight of momentum-based memory bank. The weight will gradually become closer to `gamma` in later rounds.
- `n_centroids` is the number of clusters used in region-aware sampling.
- `sample_per_group` is the number of samples selected in each high-uncertainty clusters.


# Citation 

Please cite the following paper if you are using our datasets/tool. Thanks!

```
@inproceedings{yu2022actune,
    title = "{A}c{T}une: Uncertainty-Based Active Self-Training for Active Fine-Tuning of Pretrained Language Models",
    author = "Yu, Yue and Kong, Lingkai and Zhang, Jieyu and Zhang, Rongzhi and Zhang, Chao",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.102",
    pages = "1422--1436",
}
```

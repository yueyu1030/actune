
import argparse
import os
from utils import load_and_cache_examples, load_and_cache_unlabeled_examples, init_logger, load_tokenizer
from trainer import Trainer
import torch 
import numpy as np 
import random 
import torch.nn as nn
import copy
from torch.utils.data import  ConcatDataset, TensorDataset, Subset
import json

def model_dict(model_type):
    if model_type == 'roberta-base':
        return 'roberta-base'
    elif model_type == 'bert-base':
        return 'bert-base-uncased'
    elif model_type == 'scibert':
        return 'allenai/scibert_scivocab_uncased'


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0  and torch.cuda.is_available():
        # print('yes')
        # assert 0
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    
    dev_dataset, num_labels, dev_size = load_and_cache_examples(args, tokenizer, mode="dev", size = 1000)
    test_dataset, num_labels, test_size = load_and_cache_examples(args, tokenizer, mode="test")
    try:
        train_dataset, num_labels, train_size  = load_and_cache_examples(args, tokenizer, mode= "train")
        unlabeled_dataset, unlabeled_size = load_and_cache_unlabeled_examples(args, tokenizer, mode = 'unlabeled', train_size = train_size, size = num_labels * 20000)
    except:
        unlabeled_dataset, unlabeled_size = load_and_cache_unlabeled_examples(args, tokenizer, mode = 'unlabeled', train_size = 64)
        with open(f"../datasets/{args.task}-{args.sample_labels}-0/train_idx_roberta-base_{args.al_method}_{args.sample_labels}.json", 'r') as f:
            indexes = json.load(f)
            print("number of labeled data:", len(indexes))
            train_dataset = Subset(unlabeled_dataset, indexes)
            train_size = len(indexes)


    print('number of labels:', num_labels)
    print('train_size:', train_size)
    print('dev_size:', dev_size)
    print('test_size:', test_size)
    print('unlabel_size:', unlabeled_size)
    trainer = Trainer(args, train_dataset=train_dataset, dev_dataset=dev_dataset,test_dataset=test_dataset, \
            unlabeled = unlabeled_dataset, \
            num_labels = num_labels, data_size = train_size, n_gpu = args.n_gpu
            )

    
    trainer.init_model()

    if args.method == 'active_selftrain':
        for i in range(args.rounds):
            if args.task in ['dbpedia']:
                train_sample = 100 * (i + 1)
                sample_labels = 100
            elif args.task in ['trec', 'chemprot']:
                train_sample = 50 * (i + 1)
                sample_labels = 50
                # if i == 0:
            else:
                train_sample = args.sample_labels  * (i + 1)
                sample_labels = args.sample_labels * args.n_labels
            if i == 0:
                try:
                    if 'dbpedia' in args.output_dir:
                        trainer.load_model(path = os.path.join(args.output_dir, 'model', f'checkpoint-{args.model_type}-finetune-random-train-100'))
                    elif 'trec' in args.output_dir or 'chemprot' in args.output_dir:
                        trainer.load_model(path = os.path.join(args.output_dir, 'model', f'checkpoint-{args.model_type}-finetune-random-train-100'))
                    else:
                        trainer.load_model(path = os.path.join(args.output_dir, 'model', f'checkpoint-{args.model_type}-active_selftrain-region_entropy-train-{args.sample_labels}'))
                    loss_test, acc_test = trainer.evaluate('test', 0)
                    print(f"Initial, acc={acc_test}")
                    trainer.tb_writer.add_scalar(f"FT_Test_acc_{args.method}_seed{args.seed}", acc_test,  train_sample)
                    trainer.tb_writer.add_scalar(f"ST_Test_acc_{args.method}_seed{args.seed}", acc_test,  train_sample)
                except:
                    print("Loading Error! Retrain the model.")
                    trainer.train(n_sample = train_sample)

            else:
                trainer.active_selftrain(n_sample = train_sample, soft = False)
            if args.smooth_prob == 1: # pool
                if args.pool_scheduler == 1:
                    max_sample = int(args.pool) * args.rounds
                    min_sample = int(args.pool_min)
                    sample_num = min_sample + int((max_sample - min_sample) * i/(args.rounds-1))
                else:
                    sample_num = min(int(args.pool) * (i + 1), len(trainer.unlabeled) - 1)
            elif args.pool < 1:       # 
                sample_num = int(args.pool * (len(trainer.unlabeled)- sample_labels))
            else:                     # 
                sample_num = int(args.pool)
            if sample_num < 0: # corner case, can be ignored in most cases
                sample_num = 1
                
            trainer.sample(n_sample = sample_labels, n_unlabeled = sample_num, round = i)
            query_distribution =  np.array(list(trainer.active_sampler.sample_class.values()))
            st_distribution = np.array(list(trainer.active_sampler.st_class.values()))
            trainer.tb_writer.add_histogram(f"Query_Class_Distribution", query_distribution/np.sum(query_distribution), args.sample_labels * args.n_labels * (i+1))
            trainer.tb_writer.add_histogram(f"ST_Data_Class_Distribution", st_distribution/np.sum(st_distribution) , args.sample_labels * args.n_labels * (i+1))
            trainer.reinit_model()
    
    elif args.method == 'finetune':
        for i in range(args.rounds):
            sample_num = 1
            if args.task in ['dbpedia']:
                train_sample = 100 * (i + 1)
                sample_labels = 100
            elif args.task in ['trec', 'chemprot']:
                train_sample = 50 * (i + 1)
                sample_labels = 50
            else:
                train_sample = args.sample_labels  * (i + 1)
                sample_labels = args.sample_labels * args.n_labels
            if args.task in ['trec', 'chemprot'] and i == 0: # WL init
                trainer.load_model(path = os.path.join(args.output_dir, 'model', f'checkpoint-{args.model_type}-finetune-random-train-100'))
                loss_test, acc_test = trainer.evaluate('test', 0)
            else:
                trainer.train(n_sample = train_sample)
            trainer.sample(n_sample = sample_labels, n_unlabeled = sample_num)
            trainer.reinit_model()

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='clean', type=str, help="which method to use")
    parser.add_argument("--gpu", default='0,1,2,3', type=str, help="which gpu to use")
    parser.add_argument("--n_gpu", default=1, type=int, help="which gpu to use")

    parser.add_argument("--seed", default=0, type=int, help="which seed to use")
    parser.add_argument("--task", default="agnews", type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="../datasets", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to model")
    parser.add_argument("--eval_dir", default="./eval", type=str, help="Evaluation script, result directory")
    parser.add_argument("--tsb_dir", default="./eval", type=str, help="TSB script, result directory")
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--dev_file", default="dev.tsv", type=str, help="dev file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--unlabel_file", default="unlabeled.tsv", type=str, help="Test file")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--sample_labels", default=100, type=int, help="number of labels for sampling in AL")
    parser.add_argument("--dev_labels", default=100, type=int, help="number of labels for dev set")
    parser.add_argument("--pool", default=0.1, type=float, help="number of labels for dev set")
    parser.add_argument("--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.",)

    parser.add_argument('--rounds', type=int, default=10, help="Active Learning Rounds.")
    parser.add_argument('--logging_steps', type=int, default=10, help="Log every X updates steps.")
    parser.add_argument('--tsb_logging_steps', type=int, default=10, help="Log every X updates steps.")
    parser.add_argument('--self_train_logging_steps', type=int, default=20, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")
    parser.add_argument("--model_type", default="bert-base-uncased", type=str)
    parser.add_argument("--auto_load", default=1, type=int, help="Auto loading the model or not")
    parser.add_argument("--add_sep_token", action="store_true", help="Add [SEP] token at the end of the sentence")

    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=100, type=int, help="Training steps for initialization.")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=100, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--self_training_batch_size", default=32, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--eval_batch_size", default=256, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_seq_len", default=128, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="The maximum total input sequence length after tokenization.")

    parser.add_argument("--gce_loss", default=0, type=int, help="Whether Use GCE LOSS or not.")
    parser.add_argument("--gce_loss_q", default=0.8, type=float, help="Whether Use GCE LOSS or not.")

    parser.add_argument('--self_training_max_step', type = int, default = 10000, help = 'the maximum step (usually after the first epoch) for self training')    
    parser.add_argument("--self_training_eps", default=0.6, type=float, help="The confidence thershold for the pseudo labels.")
    parser.add_argument("--self_training_power", default=2, type=float, help="The power of predictions used for self-training with soft labels.")
    parser.add_argument("--self_training_weight", default=0.5, type=float, help="The weight for self-training term.")

    parser.add_argument("--al_method", default='random', type=str, help="The initial learning rate for Adam.")

    # parser.add_argument("--balance_st", default=0, type=int, help="Balance between class.")
    # parser.add_argument("--balance_query", default=0, type=int, help="Balance between query.")
    parser.add_argument("--gamma", default=1, type=float, help="Balance between prev and current.")
    parser.add_argument("--smooth_prob", default=1, type=int, help="Balance between prev and current.")

    parser.add_argument("--n_centroids", default=25, type=int, help="Number of regions used in region-aware sampling.")
    parser.add_argument("--region_beta", default=0.1, type=float, help="The weight used in region-aware sampling.")
    parser.add_argument("--sample_per_group", default=10, type=int, help="Number of samples selected from each cluster.")
    # parser.add_argument("--region_rho", default=0.1, type=float, help="Decay weight.")
    parser.add_argument("--gamma_scheduler", default=0, type=int, help="Whether to dynamically adjust weight for momentum based memory bank.")
    parser.add_argument("--pool_scheduler", default=0, type=int, help="Whether to adjust number of unlabeled examples.")
    parser.add_argument("--gamma_min", default=0.6, type=float, help="The momentum coefficient for aggregating predictions.")
    parser.add_argument("--pool_min", default=5000, type=int, help="The minimum number of selected pseudo-labeled samples for self-training.")
    parser.add_argument("--weight_embedding", default=1, type=int, help="Whether use weighted K-means for clustering.")

    args = parser.parse_args()
    args.model_name_or_path = model_dict(args.model_type)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.task in ["SST-2"]:
        args.n_labels = 2
    elif args.task in ["agnews"]:
        args.n_labels = 4
    elif args.task in ["pubmed"]:
        args.n_labels = 5
    elif args.task in ["trec"]:
        args.n_labels = 6
    elif args.task in ["chemprot"]:
        args.n_labels = 10
    elif args.task in ["dbpedia"]:
        args.n_labels = 14
    main(args)
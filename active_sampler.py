import torch
import numpy as np
from collections import Counter
import time
from scipy import stats
import pandas as pd
from torch.nn import functional as F
from torch.utils.data.sampler import Sampler
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import faiss
from tqdm import tqdm, trange 
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans, MiniBatchKMeans
import copy             
import time


def calc_entropy(x):
    # x is the number of occurrences of each label
    lst = []
    for y in x:
        lst.append(x[y])
    lst = np.array(lst) / np.max(lst) 
    return -np.sum(lst * np.log(lst + 1e-12))

class SubsetSampler(Sampler):
    r"""Samples elements seqentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class Active_sampler(object):
    
    def __init__(self, args, train_dataset, unlabeled_dataset, seed=0):
        self.args = args
        self.npr = np.random.RandomState(seed)
        self.train_dataset = train_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.pooled_dataset = None
        # self.get_sample = {
        #                     'random': self.get_random,
        #                     'entropy': self.get_max_entropy,
        #                     'cal': self.get_cal, 
        #                 }
        self.sample_class = Counter()
        self.st_class = Counter()
        self.mem = []

    def convert_tensor_to_dataset(self, tensor, prediction = None):
        if prediction is None:
            return TensorDataset(tensor[0],tensor[1], tensor[2],tensor[3],tensor[4],)
        else:
            prediction = torch.FloatTensor(prediction)
            # print(tensor[0].shape,tensor[1].shape, tensor[2].shape,tensor[3].shape,tensor[4].shape, prediction.shape)
            return TensorDataset(tensor[0],tensor[1], tensor[2],tensor[3],tensor[4], prediction)

    def sample(self, method, train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, entropy = None, n_sample = 100, n_unlabeled = 2048, round = 1):
        print(f"Active sampling: {method}， Samping {n_sample} data, add {n_unlabeled} to pool in total!")
        self.train_pred = train_pred
        self.train_feat = train_feat
        self.train_label = train_label
        self.unlabeled_pred = unlabeled_pred
        self.unlabeled_feat = unlabeled_feat
        self.unlabeled_label = unlabeled_label
        self.unlabel_pseudo = np.argmax(unlabeled_pred, axis = -1)
        self.unlabel_correct = (self.unlabel_pseudo == unlabeled_label).astype(int)
        len_unlabel = unlabeled_pred.shape[0]
        if method == 'cal' and self.args.smooth_prob == 1 and len(self.mem) > 0:
            unlabeled_pred = (1 - self.args.gamma) * self.mem + self.args.gamma * self.unlabeled_pred
        
        if method == 'random':
            idx = np.random.permutation(len_unlabel)
            value = np.sum(-np.log(unlabeled_pred + 1e-12) * unlabeled_pred, axis = -1)
        elif method == 'entropy':
            idx, value = self.get_max_entropy(unlabeled_pred, n_sample, n_unlabeled)
        elif method == 'cal':
            idx, value = self.get_cal(train_pred, train_feat,  unlabeled_pred, unlabeled_feat, n_sample, n_unlabel = n_unlabeled)
        elif method == 'region_cal':
            idx, value = self.get_region_cal(train_pred, train_feat,  unlabeled_pred, unlabeled_feat, n_sample, n_unlabel = n_unlabeled, ncentroids = self.args.n_centroids, sample_per_group=self.args.sample_per_group, beta = self.args.region_beta, weight = self.args.weight_embedding)
        elif method == 'region_entropy':
            idx, value = self.get_region_entropy(train_pred, train_feat,  unlabeled_pred, unlabeled_feat, n_sample, n_unlabel = n_unlabeled, ncentroids = self.args.n_centroids, sample_per_group=self.args.sample_per_group, beta = self.args.region_beta, weight = self.args.weight_embedding)
        
        if len(self.mem) == 0:
            if self.args.smooth_prob == 1:
                self.mem = self.unlabeled_pred
            else:
                self.mem = value
        else:
            if self.args.gamma_scheduler == 1: # gradually upweight gamma in AL rounds
                gamma = self.args.gamma_min + (self.args.gamma - self.args.gamma_min) * ((round-1) / (self.args.rounds-2))
            else:
                gamma = self.args.gamma
            print("Gamma", gamma)
            if self.args.smooth_prob == 1:
                self.mem = (1 - gamma) * self.mem + gamma * self.unlabeled_pred
            else:
                self.mem = (1 - gamma) * self.mem + gamma * value
        idx = list(idx)
        sample_idx = idx[:n_sample]
        if self.args.smooth_prob == 1:
            save_idx = idx[n_sample:]
            smooth_val = np.sum(-np.log(self.mem + 1e-12) * self.mem, axis = -1)
            smooth_idx = list(np.argsort(smooth_val))[::-1]
        else:
            smooth_idx = list(np.argsort(self.mem))[::-1]
            save_idx = idx[n_sample : -n_unlabeled]
        pool_idx = smooth_idx[-n_unlabeled:]
        pool_idx = pool_idx[::-1]

        indexes = np.arange(len(idx))     
        n_class = unlabeled_pred.shape[-1]   
        pool_idx_class = []
        save_idx_class = []
        sample_idx_class = []
        for i in range(n_class):
            label_idx = (self.unlabel_pseudo == i)
            if self.args.smooth_prob == 1:
                value_class = value[label_idx]
            else:
                value_class = self.mem[label_idx]
            index_class = indexes[label_idx]

            class_idx = np.argsort(value_class)[::-1]
            sorted_index = index_class[class_idx]

            pool_idx_tmp = list(sorted_index[-n_unlabeled//n_class:])
            sample_idx_tmp = list(sorted_index[:n_sample//n_class])
            save_idx_tmp = list(sorted_index[n_sample//n_class:-n_unlabeled//n_class])
            pool_idx_class += pool_idx_tmp
            save_idx_class += save_idx_tmp
            sample_idx_class += sample_idx_tmp

        # if self.args.balance_st:
        #     pool_idx = pool_idx_class
        # if self.args.balance_query:
        #     sample_idx = sample_idx_class
        items = {}
        for x in sample_idx:
            items[x] = 1
        
        pool_idx = list( set(pool_idx) - (set(pool_idx) & set(sample_idx)) )
        if self.args.smooth_prob == 1:
            for x in sample_idx:
                items[x] = 1
        else:
            for x in pool_idx:
                items[x] = 1
        # if self.args.balance_st or self.args.balance_query:
        #     save_idx = [i for i in range(len(idx)) if i not in items] 
        self.mem = self.mem[save_idx]
        print(self.mem.shape)
        sample_dataset = self.convert_tensor_to_dataset(self.unlabeled_dataset[sample_idx])
        unlabeled_dataset = self.convert_tensor_to_dataset(self.unlabeled_dataset[save_idx])
        pooled_dataset = self.convert_tensor_to_dataset(self.unlabeled_dataset[pool_idx], unlabeled_pred[pool_idx])

        train_dataset = ConcatDataset([self.train_dataset, sample_dataset])
        self.train_dataset = train_dataset
        
        if self.args.smooth_prob == 1:
            self.pooled_dataset = pooled_dataset
            self.unlabeled_dataset = unlabeled_dataset
        else:
            self.unlabeled_dataset = unlabeled_dataset
            if self.pooled_dataset:
                self.pooled_dataset  = ConcatDataset([self.pooled_dataset, pooled_dataset])
            else:
                self.pooled_dataset = pooled_dataset
        self.sample_class.update(unlabeled_label[sample_idx])
        self.st_class.update(np.argmax(unlabeled_pred[pool_idx], axis = -1))
        return self.train_dataset, self.unlabeled_dataset, self.pooled_dataset 

    def get_random(self, unlabeled_pred, n_sample):
        entropy = np.sum(-np.log(unlabeled_pred + 1e-12) * unlabeled_pred, axis = -1)

        len_unlabel = unlabeled_pred.shape[0]
        rand_idx = np.random.permutation(len_unlabel)
        return rand_idx, entropy

    def get_max_entropy(self, unlabeled_pred, n_sample, n_unlabel = 2048):
        entropy = np.sum(-np.log(unlabeled_pred + 1e-12) * unlabeled_pred, axis = -1)
        idx = np.argsort(entropy)[::-1]
        return idx, entropy

    def get_cal(self, train_pred, train_feat,  unlabeled_pred, unlabeled_feat, n_sample, n_unlabel, k = 10):
        d = train_feat.shape[-1]
        index = faiss.IndexFlatL2(d)
        index.add(train_feat)
        D, I = index.search(unlabeled_feat, k)
        # print(I.shape)
        # print(train_pred[I].shape)
        # print(unlabeled_pred.shape)
        unlabeled_pred =  np.expand_dims(unlabeled_pred, axis = 1)
        # print(unlabeled_pred.shape)
        score = np.log((1e-10+train_pred[I])/ (1e-10+unlabeled_pred)) * train_pred[I]
        # print(score.shape)
        mean_kl = np.mean(np.sum(score, axis = -1), axis = -1)
        idx = np.argsort(mean_kl)[::-1]
        sample_idx = list(idx[:n_sample])
        save_idx = list(idx[n_sample:])
        sample_dataset = self.convert_tensor_to_dataset(self.unlabeled_dataset[sample_idx])
        unlabeled_dataset = self.convert_tensor_to_dataset(self.unlabeled_dataset[save_idx])
        train_dataset = ConcatDataset([self.train_dataset, sample_dataset])
        # self.train_dataset = train_dataset
        # self.unlabeled_dataset = unlabeled_dataset
        return idx, mean_kl

    
    def get_region_cal(self, train_pred, train_feat,  unlabeled_pred, unlabeled_feat, n_sample, n_unlabel, ncentroids = 25, sample_per_group=10, beta = 1, k = 10, weight = True):
        d = train_feat.shape[-1]
        index = faiss.IndexFlatL2(d)
        index.add(train_feat)
        D, I = index.search(unlabeled_feat, k)
        unlabeled_pred_expand =  np.expand_dims(unlabeled_pred, axis = 1)
        score = np.log((1e-10+train_pred[I])/ (1e-10+unlabeled_pred_expand)) * train_pred[I]
        entropy = np.mean(np.sum(score, axis = -1), axis = -1)

        d = unlabeled_feat.shape[-1]
        if weight:
            kmeans = MiniBatchKMeans(n_clusters = ncentroids, random_state=0, batch_size=256, n_init=3, max_iter=100) 
            kmeans.fit(unlabeled_feat, sample_weight = copy.deepcopy(entropy))
            index = faiss.IndexFlatL2(d)
            index.add(kmeans.cluster_centers_)
            D, I = index.search(unlabeled_feat, 1)
        else:
            kmeans = faiss.Clustering(int(d), ncentroids)
            index = faiss.IndexFlatL2(d)
            kmeans.train(unlabeled_feat, index)
            centroid = faiss.vector_to_array(kmeans.centroids).reshape(ncentroids, -1)
            index.add(centroid)
            D, I = index.search(unlabeled_feat, 1)
        I = I.flatten()
        unlabeled_pseudo = np.argmax(unlabeled_pred, axis = 1)
        scores = []
        indexes = []
        for i in range(ncentroids):
            idx = (I == i)
            cnt = Counter()
            mean_entropy = np.mean(entropy[idx])
            for z in unlabeled_pseudo[idx]:
                cnt[z] += 1
            class_entropy = calc_entropy(cnt)
            value = mean_entropy + beta * class_entropy
            scores.append(value)
            sorted_idx = np.argsort(entropy[idx]) 
            idxs = np.arange(len(I))[idx][sorted_idx]                     
            indexes.append(list(idxs))
        sample_idx = []
        remains = n_sample
        for i in np.argsort(scores)[::-1]:
            if self.args.task == "SST-2":
                topK = 10
            else:
                topK = 20
            sample_idx += indexes[i][-min(remains, sample_per_group, len(indexes[i])//topK):]
            indexes[i] = indexes[i][:-min(remains, sample_per_group, len(indexes[i])//topK)]
            remains -= len( indexes[i][-min(remains, sample_per_group, len(indexes[i])//topK):])
            if remains <= 0:
                break 
        for y in indexes:
            sample_idx += y
        return sample_idx, entropy

    def get_region_entropy(self, train_pred, train_feat,  unlabeled_pred, unlabeled_feat, n_sample, n_unlabel, ncentroids = 25, sample_per_group=10, beta = 1, weight = True):
        entropy = np.sum(-np.log(unlabeled_pred + 1e-12) * unlabeled_pred, axis = -1)
        d = unlabeled_feat.shape[-1]
        if weight: # use weighted K-Means Clustering
            kmeans = MiniBatchKMeans(n_clusters = ncentroids, random_state=0, batch_size=256, n_init=3, max_iter=100) 
            kmeans.fit(unlabeled_feat, sample_weight = copy.deepcopy(entropy))
            index = faiss.IndexFlatL2(d)
            index.add(kmeans.cluster_centers_)
            D, I = index.search(unlabeled_feat, 1)
        else:
            kmeans = faiss.Clustering(int(d), ncentroids)
            index = faiss.IndexFlatL2(d)
            kmeans.train(unlabeled_feat, index)
            centroid = faiss.vector_to_array(kmeans.centroids).reshape(ncentroids, -1)
            index.add(centroid)
            D, I = index.search(unlabeled_feat, 1)
        I = I.flatten()
        unlabeled_pseudo = np.argmax(unlabeled_pred, axis = 1)
        scores = []
        indexes = []
        for i in range(ncentroids):
            idx = (I == i)
            cnt = Counter()
            # calculate the mean entropy of samples
            mean_entropy = np.mean(entropy[idx])
            for z in unlabeled_pseudo[idx]:
                cnt[z] += 1
            # calculate the mean entropy of pseudo labels
            class_entropy = calc_entropy(cnt)
            value = mean_entropy + beta * class_entropy
            scores.append(value)
            sorted_idx = np.argsort(entropy[idx]) 
            idxs = np.arange(len(I))[idx][sorted_idx]                     
            indexes.append(list(idxs))
        sample_idx = []
        remains = n_sample
        for i in np.argsort(scores)[::-1]:
            if self.args.task == "SST-2":
                topK = 10
            else:
                topK = 20
            sample_idx += indexes[i][-min(remains, sample_per_group, len(indexes[i])//topK):]
            indexes[i] = indexes[i][:-min(remains, sample_per_group, len(indexes[i])//topK)]
            remains -= len( indexes[i][-min(remains, sample_per_group, len(indexes[i])//topK):])
            if remains <= 0:
                break 
        for y in indexes:
            sample_idx += y
        return sample_idx, entropy

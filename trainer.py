import os
import logging
from tqdm import tqdm, trange
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset, TensorDataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup, AutoConfig, AutoModelForSequenceClassification
import copy
import math
import os
import random 
from active_sampler import Active_sampler
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

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

def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)

def acc_and_f1(preds, labels, average='macro'):
    acc = (preds == labels).mean()

    return {
        "acc": acc,
    }


class Trainer(object):
    def __init__(self, args, train_dataset = None, dev_dataset = None, test_dataset = None, unlabeled = None, \
                num_labels = 10, data_size = 100, n_gpu = 1):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.unlabeled = unlabeled
        self.data_size = data_size

        self.num_labels = num_labels
        self.config_class = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=self.num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=self.num_labels)
        self.n_gpu = n_gpu
        self.tb_writer = SummaryWriter(self.args.tsb_dir)
        self.active_sampler = Active_sampler(args = self.args, train_dataset = self.train_dataset, unlabeled_dataset = self.unlabeled)
        
    def soft_frequency(self, logits, soft = True):
        """
        Unsupervised Deep Embedding for Clustering Analysis
        https://arxiv.org/abs/1511.06335
        """
        power = self.args.self_training_power
        y = logits
        f = torch.sum(y, dim=0)
        t = y**power / f
        t = t + 1e-10
        p = t/torch.sum(t, dim=-1, keepdim=True)
        return p if soft else torch.argmax(p, dim=1)

    def calc_loss(self, input, target, loss, thresh = 0.5, soft = True, conf = None, is_prob = False):
        softmax = nn.Softmax(dim=1)
        if not is_prob:
            target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
        
        if conf == 'max':
            weight = torch.max(target, axis = 1).values
            w = torch.FloatTensor([1 if x == True else 0 for x in weight>thresh]).to(target.device)
        elif conf == 'entropy':
            weight = torch.sum(-torch.log(target+1e-6) * target, dim = 1)
            weight = 1 - weight / np.log(weight.size(-1))
            w = torch.FloatTensor([1 if x == True else 0 for x in weight>thresh]).to(target.device)
        elif conf is None:
            weight = torch.ones(target.shape[0]).to(target.device)
            w =  torch.ones(target.shape[0]).to(target.device)
            
        target = self.soft_frequency(target, soft = soft)
        loss_batch = loss(input, target)
        l = loss_batch * w.unsqueeze(1) * weight.unsqueeze(1)
        return l, weight, w 
    
    def gce_loss(self, input, target, thresh = 0.5, soft = True, conf = None, is_prob = False):
        softmax = nn.Softmax(dim=1)
        if not is_prob:
            target = softmax(target.view(-1, target.shape[-1])).view(target.shape)
        weight = torch.max(target, axis = 1).values
        target = torch.argmax(target, dim = -1)
        if self.args.gce_loss_q == 0:
            if input.size(-1) == 1:
                ce_loss = nn.BCEWithLogitsLoss(reduction='none')
                loss = ce_loss(input.view(-1), input.float())
            else:
                ce_loss = nn.CrossEntropyLoss(reduction='none')
                loss = ce_loss(input, target)
        else:
            if input.size(-1) == 1:
                pred = torch.sigmoid(input)
                pred = torch.cat((1-pred, pred), dim=-1)
            else:
                pred = F.softmax(input, dim=-1)
            pred_ = torch.gather(pred, dim=-1, index=torch.unsqueeze(target, -1))
            w = pred_ > thresh
            loss = (1 - pred_ ** self.args.gce_loss_q) / self.args.gce_loss_q
            loss = (loss * w)        
        return loss, weight, w 

    def init_model(self):
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and self.n_gpu > 0 else "cpu"
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)
    
    def load_model(self, path = None):
        if path is None:
            logger.info("No ckpt path, load from original ckpt!")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name_or_path,
                config=self.config_class,
                cache_dir=self.args.cache_dir if self.args.cache_dir else None,
            ).to(self.device)
        else:
            logger.info(f"Loading from {path}!")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                path,
                config=self.config_class,
                cache_dir=self.args.cache_dir if self.args.cache_dir else None,
            ).to(self.device)
        self.init_model()

    def reinit_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
                self.args.model_name_or_path,
                config=self.config_class,
                cache_dir=self.args.cache_dir if self.args.cache_dir else None,
            ).to(self.device)
        self.init_model()

    def save_dataset(self, stage = 0):
        output_dir = os.path.join(
            self.args.output_dir, "dataset", "dataset-{}-{}-{}-{}".format(self.args.model_type, self.args.method, self.args.al_method, stage))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.train_dataset, os.path.join(output_dir, 'train'))
        torch.save(self.dev_dataset, os.path.join(output_dir, 'dev'))
        torch.save(self.test_dataset, os.path.join(output_dir, 'test'))
        torch.save(self.unlabeled, os.path.join(output_dir, 'unlabeled'))
        if self.pooled:
            torch.save(self.unlabeled, os.path.join(output_dir, 'pooled'))


    def load_dataset(self, stage = 0):
        load_dir = os.path.join(
            self.args.output_dir, "dataset", "dataset-{}-{}-{}-{}".format(self.args.model_type, self.args.method, self.args.al_method, stage))
        if not os.path.exists(load_dir):
            # except:
            load_dir = os.path.join(
                self.args.output_dir, "dataset", "dataset-{}-{}-{}".format(self.args.model_type, self.args.al_method, stage))
        self.train_dataset = torch.load(os.path.join(load_dir, 'train'))
        self.dev_dataset = torch.load(os.path.join(load_dir, 'dev'))
        self.test_dataset = torch.load(os.path.join(load_dir, 'test'))
        self.unlabeled = torch.load(os.path.join(load_dir, 'unlabeled'))
        

    def save_result(self, stage = 0, acc = 0, self_training = False):
        if self_training:
            setup = 'self_training'
        else:
            setup = 'train'
        output_dir = os.path.join(
            self.args.output_dir, "result", "result-{}-{}-{}-{}-{}".format(self.args.model_type,self.args.method, self.args.al_method, setup,  stage))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, 'acc.json') , 'w') as f:
            json.dump({"acc": acc, "stage": stage, "method": self.args.method, "model_type":self.args.model_type, "al_method": self.args.al_method}, f)

    def save_model(self, stage = 0, self_training = False):
        if self_training:
            setup = 'self_training'
        else:
            setup = 'train'
        output_dir = os.path.join(
            self.args.output_dir, "model", "checkpoint-{}-{}-{}-{}-{}".format(self.args.model_type,self.args.method, self.args.al_method, setup, stage))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        # torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
        logger.info("Saving model checkpoint to %s", output_dir)


    def train(self, n_sample = 20):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        training_steps = int(self.args.num_train_epochs) * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = int(training_steps * 0.05), num_training_steps = training_steps)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", training_steps)
        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args)
        criterion = nn.CrossEntropyLoss(reduction = 'mean')
        best_model = None
        best_dev = -np.float('inf')
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                          }
                outputs = self.model(**inputs)
                loss = outputs[0]
                logits = outputs[1]
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                if torch.cuda.device_count() > 1:
                    loss = loss.mean()
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    epoch_iterator.set_description("iteration:%d, Loss:%.3f, best dev:%.3f" % (_, tr_loss/global_step, 100*best_dev))
                    if (self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0) or (step in [len(train_dataloader)//2, len(train_dataloader)//4]):
                        loss_dev, acc_dev = self.evaluate('dev', global_step)
                        self.tb_writer.add_scalar(f"FT_Dev_acc_sample{n_sample}", acc_dev, global_step)
                        if acc_dev > best_dev:
                            logger.info("Best model updated!")
                            self.best_model = copy.deepcopy(self.model.state_dict())
                            best_dev = acc_dev
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model(stage = n_sample)
                
                if 0 < training_steps < global_step:
                    epoch_iterator.close()
                    break
            loss_dev, acc_dev = self.evaluate('dev', global_step)
            print(f'Dev: Loss: {loss_dev}, Acc: {acc_dev}')
            self.tb_writer.add_scalar(f"FT_Dev_acc_sample{n_sample}", acc_dev, global_step)
            if acc_dev > best_dev:
                logger.info("Best model updated!")
                self.best_model = copy.deepcopy(self.model.state_dict())
                best_dev = acc_dev            
        self.model.load_state_dict(self.best_model)
        loss_test, acc_test = self.evaluate('test', global_step)
        print(f'Test: Loss: {loss_test}, Acc: {acc_test}')
        self.tb_writer.add_scalar(f"FT_Test_acc_{self.args.method}_seed{self.args.seed}", acc_test, n_sample)
        self.save_model(stage = n_sample)
        self.save_result(stage = n_sample, acc = acc_test, self_training = False)
        return global_step, tr_loss / global_step
     
    def active_selftrain(self, soft = True, n_sample = 50):
        train_sampler = RandomSampler(self.train_dataset) 
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)
        train_dataloader_iter = iter(train_dataloader)
        unlabeled_sampler = RandomSampler(self.pooled)
        unlabeled_dataloader = DataLoader(self.pooled, sampler=unlabeled_sampler, batch_size=self.args.self_training_batch_size)
        unlabeled_dataloader_iter = iter(unlabeled_dataloader)
        if self.args.self_training_max_step > 0:
            t_total = self.args.self_training_max_step
            self.args.num_train_epochs = self.args.self_training_max_step // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate * 0.25, eps=self.args.adam_epsilon)
        self_training_loss = nn.KLDivLoss(reduction = 'none') if soft else nn.CrossEntropyLoss(reduction = 'none')
        softmax = nn.Softmax(dim=1)
        update_step = 0
        self_training_steps = self.args.self_training_max_step
        global_step = 0
        selftrain_loss = 0
        best_model = None
        best_dev = -np.float('inf')
        set_seed(self.args)
        step_iterator = trange(int(self_training_steps * self.args.gradient_accumulation_steps))
        for step in step_iterator:
            # epoch_iterator = tqdm(train_dataloader, desc="SelfTrain, Iteration")
            try:
                batch = next(train_dataloader_iter)
            except StopIteration:
                logger.info("Finished iterating Train dataset, begin reiterate")
                train_dataloader_iter = iter(train_dataloader)
                batch = next(train_dataloader_iter)
            try:
                batch_unlabeled = next(unlabeled_dataloader_iter)
            except StopIteration:
                logger.info("Finished iterating Unlabeled dataset, begin reiterate")
                unlabeled_dataloader_iter = iter(unlabeled_dataloader)
                batch_unlabeled = next(unlabeled_dataloader_iter)
            self.model.train()
            batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU     
            inputs_train = {
                        'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2],
                        'labels': batch[3],
                        'output_hidden_states':True
                    }
            
            batch_unlabeled = tuple(t.to(self.device) for t in batch_unlabeled)  # GPU or CPU     
            inputs_unlabeled = {
                        'input_ids': batch_unlabeled[0],
                        'attention_mask': batch_unlabeled[1],
                        'token_type_ids': batch_unlabeled[2],
                        'labels': batch_unlabeled[3], # Never use this!
                        "output_hidden_states": True
                    }
            outputs_train = self.model(**inputs_train)
            outputs = self.model(**inputs_unlabeled)
            outputs_pseudo = batch_unlabeled[-1]
            logits = outputs[1]
            if self.args.gce_loss: # an alternative for denoising function, that can further boost the performance :) We do not use it in our main experiments.
                loss_st, weight, w = self.gce_loss(input = logits, \
                                target= outputs_pseudo, \
                                thresh = self.args.self_training_eps, \
                                soft = soft, \
                                conf = 'max', \
                                is_prob = True)
            else:
                loss_st, weight, w = self.calc_loss(input = torch.log(softmax(logits)), \
                                target= outputs_pseudo, \
                                loss = self_training_loss, \
                                thresh = self.args.self_training_eps, \
                                soft = False, \
                                conf = 'max', \
                                is_prob = True)
            weight = weight.unsqueeze(1).detach().cpu().numpy()
            w = w.flatten().bool().detach().cpu().numpy()
           
            train_loss = outputs_train[0]
            if torch.cuda.device_count() > 1:
                train_loss = train_loss.mean()
            loss_st = loss_st.mean()
            loss = (1 - self.args.self_training_weight) * train_loss +  self.args.self_training_weight * loss_st
            clean_loss = train_loss.item()
            selftrain_loss = loss_st.item()
            all_loss = loss.item()
            loss.backward()            
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                self.model.zero_grad()
                global_step += 1
                step_iterator.set_description("Active SelfTrain iter:%d Loss:%.3f, weight: %.2f, Clean loss: %.3f, selftrain loss: %.3f" % (step, all_loss, self.args.self_training_weight, clean_loss, selftrain_loss))

                if global_step % self.args.self_train_logging_steps == 0:
                    loss_dev, acc_dev = self.evaluate('dev', global_step)
                    self.tb_writer.add_scalar(f"ST_Acc_Dev_sample{n_sample}", acc_dev, step)
                    print(f'Stage 1, Dev: Loss: {loss_dev}, Acc: {acc_dev}')
                    if acc_dev > best_dev:
                        logger.info("Best model updated!")
                        best_model = copy.deepcopy(self.model.state_dict())
                        best_dev = acc_dev
        
        self.model.load_state_dict(best_model)
        loss_test, acc_test = self.evaluate('test', global_step)
        print(f'Test: Loss: {loss_test}, Acc: {acc_test}')
        self.tb_writer.add_scalar(f"ST_Test_acc_{self.args.method}_seed{self.args.seed}", acc_test, n_sample)
        self.save_model(stage = n_sample, self_training = True)
        self.save_result(stage = n_sample, acc = acc_test, self_training = True)

    def sample(self, n_sample = 20, n_unlabeled = 2048, round = 1):
        train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_logits = self.inference(layer = -1)

        new_train, new_unlabeled, pooled = self.active_sampler.sample(self.args.al_method, train_pred, train_feat, train_label,  unlabeled_pred, \
            unlabeled_feat, unlabeled_label, n_sample= n_sample, n_unlabeled = n_unlabeled, round = round)
        self.train_dataset = new_train
        self.unlabeled = new_unlabeled
        self.pooled = pooled
        print(f"=======  train {len(new_train)}, unlabel {len(new_unlabeled)} pool {len(pooled)}  =========")
        self.save_dataset(stage = n_sample)
        return new_train, new_unlabeled

    
    def inference(self, layer = -1):
        ## Inference the embeddings/predictions for unlabeled data
        train_dataloader = DataLoader(self.train_dataset, shuffle=False, batch_size=self.args.eval_batch_size)
        train_pred = []
        
        train_feat = []
        train_label = []
        self.model.eval()
        softmax = nn.Softmax(dim = 1)
        for batch in tqdm(train_dataloader, desc="Evaluating Labeled Set"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                            'output_hidden_states': True
                          }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits, feats = outputs[0], outputs[1], outputs[2]
                # print(outputs)
                logits = softmax(logits).detach().cpu().numpy()
                train_pred.append(logits)
                train_feat.append(feats[layer][:, 0, :].detach().cpu().numpy())
                train_label.append(batch[3].detach().cpu().numpy())
        train_pred = np.concatenate(train_pred, axis = 0)
        train_feat = np.concatenate(train_feat, axis = 0)
        train_label = np.concatenate(train_label, axis = 0)
        train_conf = np.amax(train_pred, axis = 1)
        print("train size:", train_pred.shape, train_feat.shape, train_label.shape, train_conf.shape)
        unlabeled_dataloader = DataLoader(self.unlabeled, shuffle=False, batch_size=self.args.eval_batch_size)
        unlabeled_pred = []
        unlabeled_logits = []
        unlabeled_feat = []
        unlabeled_label = []
        self.model.eval()
        for batch in tqdm(unlabeled_dataloader, desc="Evaluating Unlabeled Set"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                            'output_hidden_states': True
                          }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits, feats = outputs[0], outputs[1], outputs[2]
                unlabeled_logits.append(logits.detach().cpu().numpy())
                logits = softmax(logits).detach().cpu().numpy()
                unlabeled_pred.append(logits)
                unlabeled_feat.append(feats[layer][:, 0, :].detach().cpu().numpy())
                unlabeled_label.append(batch[3].detach().cpu().numpy())
        unlabeled_feat = np.concatenate(unlabeled_feat, axis = 0)
        unlabeled_label = np.concatenate(unlabeled_label, axis = 0)
        unlabeled_pred = np.concatenate(unlabeled_pred, axis = 0)
        unlabeled_logits = np.concatenate(unlabeled_logits, axis = 0)
        unlabeled_conf = np.amax(unlabeled_pred, axis = 1)
        unlabeled_pseudo = np.argmax(unlabeled_pred, axis = 1)
        
        print("unlabeled size:", unlabeled_pred.shape, unlabeled_feat.shape, unlabeled_label.shape, unlabeled_conf.shape)
        return train_pred, train_feat, train_label,  unlabeled_pred, unlabeled_feat, unlabeled_label, unlabeled_logits



    def evaluate(self, mode, global_step=-1):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        # logger.info("  Batch size = %d", self.args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        self.model.eval()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                            'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3],
                          }
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        result.update(result)

        logger.info("***** Eval results *****")
  
        return results["loss"], result["acc"]
    
    
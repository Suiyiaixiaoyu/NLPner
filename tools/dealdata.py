import torch
from torch.utils.data import Dataset
import random
import numpy as np
import os

class dizhidataset(Dataset):
    def __init__(self,examples,tokenizer,args,label_list):
        super(dizhidataset, self).__init__()
        self.examples = examples
        self.args = args
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.dataset = dizhidataset.makedataset(self.examples,self.tokenizer,self.args,self.label_list)

    @staticmethod
    def makedataset(examples,tokenizer,args,label_list):
        label2id = {label:id for id,label in enumerate(label_list)}
        features_token = []
        features_mask = []
        features_start_ids = []
        features_end_ids = []
        features_subjects_id = []
        for (ex_index,example) in enumerate(examples):
            textlist = example.text_a
            subjects = example.subject
            if isinstance(textlist,list):
                textlist = " ".join(textlist)
            tokens = tokenizer(textlist,return_tensors='pt').input_ids[0]
            attention_mask = tokenizer(textlist,return_tensors='pt').attention_mask[0]
            start_ids = [0]*(len(tokens)-2)
            end_ids = [0]*(len(tokens)-2)
            subjects_id = []

            for subject in subjects:
                label = subject[0]
                start = subject[1]
                end = subject[2]
                start_ids[start] = label2id[label]
                end_ids[end] = label2id[label]
                subjects_id.append((label2id[label],start,end))

            start_ids += [0]
            start_ids = [0]+start_ids
            end_ids += [0]
            end_ids = [0]+end_ids

            features_token.append(tokens)
            features_mask.append(attention_mask)
            features_start_ids.append(start_ids)
            features_end_ids.append(end_ids)
            features_subjects_id.append(subjects_id)
        dataset = list(zip(features_token,features_mask,features_start_ids,features_end_ids,features_subjects_id))
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        token,attention_mask,start_ids,end_ids,subjects_id = self.dataset[idx]
        return {'token':token,'attention_mask':attention_mask,'start_ids':start_ids,'end_ids':end_ids,'subjects_id':subjects_id}

class PadBatchSeq:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        res = dict()
        max_len = max([len(i['token']) for i in batch])
        res['token'] = torch.LongTensor([i['token'].numpy().tolist() + [self.pad_id] * (max_len - len(i['token'])) for i in batch])
        res['attention_mask'] = torch.LongTensor([i['attention_mask'].numpy().tolist() + [self.pad_id] * (max_len - len(i['attention_mask'])) for i in batch])
        res['start_ids'] = torch.LongTensor(
            [i['start_ids'] + [self.pad_id] * (max_len - len(i['start_ids'])) for i in
             batch])
        res['end_ids'] = torch.LongTensor(
            [i['end_ids'] + [self.pad_id] * (max_len - len(i['end_ids'])) for i in
             batch])
        res['subjects_id'] = [i['subjects_id'] for i in batch]
        return res

def set_seed(seed = 427):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
















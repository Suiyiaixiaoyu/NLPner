import csv
import json
import torch
import copy
import os
import re
from transformers import BertTokenizer



def read_test(input_file):
    lines = []
    with open(input_file,'r',encoding='utf-8') as f:
        for line in f:
            x = line.split('\x01')[1].strip()
            lines.append(x)

    return lines
class InputExample(object):
    def __init__(self,guid,text_a,subject):
        self.guid = guid
        self.text_a = text_a
        self.subject = subject

    def __repr__(self):
        # return {'guid':self.guid,'text_a':self.text_a,'subject':self.subject}
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(),ensure_ascii=False)+"\n"


class DataProcessor(object):

    def get_train_examples(self,data_dir):
        raise NotImplementedError()

    def get_dev_examples(self,data_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_text(self,input_file):
        lines = []
        with open(input_file,'r',encoding='utf-8') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words":words,"labels":labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    if bool(re.search(r'[^\uf881]', splits[0])):
                        words.append(splits[0])
                        if len(splits) > 1:
                            labels.append(splits[-1].replace("\n", ""))
                        else:
                            labels.append("O")
                    else:
                        continue
            if words:
                lines.append({"words":words,"labels":labels})
        return lines

def get_entity_bios(seq,id2label):
    chunks = []
    chunk = [-1,-1,-1]

    for indx,tag in enumerate(seq):
        if not isinstance(tag,str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1,-1,-1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1,-1,-1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1,-1,-1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]

        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1,-1,-1]
    return chunks

class nerProcessor(DataProcessor):
    def get_train_examples(self,data_dir):
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.conll")),"train")

    def get_dev_examples(self,data_dir):
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.conll")), "dev")

    def get_test_examples(self,data_dir):
        return self._create_examples(self._read_text(os.path.join(data_dir, "final_test.txt")), "dev")

    def get_labels(self):

        return ['O','prov','city','district','devzone','town','community','village_group',
                'road','roadno','poi','subpoi','houseno','cellno','floorno','roomno',
                'detail','assist','distance','intersection','reedundant']


    def _create_examples(self,lines,set_type):
        examples = []
        for (i,line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type,i)
            text_a = line['words']
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-','I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            subject = get_entity_bios(labels,id2label=None)
            examples.append(InputExample(guid = guid,text_a = text_a,subject = subject))
        return examples





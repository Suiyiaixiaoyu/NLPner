from precessors.ner_span import read_test
from transformers import BertTokenizerFast
from tools.args_tool import getparse
import torch
from precessors.ner_span import nerProcessor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm

args = getparse().parse_args()
device = torch.device(f'cuda:{args.GPUNUM}') if torch.cuda.is_available() else torch.device('cpu')
test = read_test('data/final_test.txt')

process = nerProcessor()
labellist = process.get_labels()
id2label = {i: label for i, label in enumerate(labellist)}
label2id = {label: i for i, label in enumerate(labellist)}

tokenizer = BertTokenizerFast.from_pretrained(args.bert_path)

pbr = tqdm(test)
model = torch.load('new_model.pth')
model = model.to(device)
model.eval()
preds = []
for text in pbr:
    n_text = []
    for char in text:
        n_text.append(char)
    n_text += ['SEP']
    n_text = ['CLS'] + n_text
    n_text = tokenizer.convert_tokens_to_ids(n_text)
    # token = text.input_ids.to(device)
    token = torch.tensor(n_text).to(device).unsqueeze(0)
    # attention_mask = text.attention_mask.to(device)
    attention_mask = torch.tensor([1]*len(n_text)).to(device).unsqueeze(0)
    start_logits, end_logits = model(token,attention_mask)
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]
    S = [0 for _ in range(len(start_pred))]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                if i == j+i:
                    S[i] = "S-"+id2label[s_l]
                    break
                else:
                    S[i] = "B-"+id2label[s_l]
                    if i+1 != j+i:
                        for index in range(i+1,j+i):
                            S[index] = "I-"+id2label[s_l]
                    S[j+i] = "E-" + id2label[s_l]
                    break
            else:
                continue
    T = S.copy()
    for ss,x in enumerate(T):
        if x == 0:
            S[ss] = 'O'
    preds.append(" ".join(S))
lines = []
with open('data/final_test.txt','r',encoding='utf-8') as f:
    for line in f:
        lines.append(line.strip())

f = open('柯哀无敌_ addr_parsing_runid.txt', 'w', encoding='utf-8')
for index,line in enumerate(lines):
    f.write(line+'\x01'+preds[index]+'\n')

f.close()












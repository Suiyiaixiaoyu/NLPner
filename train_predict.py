import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.dealdata import set_seed
from tools.args_tool import getparse
from precessors.ner_span import nerProcessor
from tools.dealdata import dizhidataset,PadBatchSeq
from transformers import BertTokenizerFast,AdamW,get_linear_schedule_with_warmup,BertConfig
from torch.utils.data import DataLoader
from models.bertforner import BertSpanForNer
from apex import amp
from progressbar import ProgressBar
from metrics.ner_metrics import SpanEntityScore,bert_extract_item
from adv.adversarial import PGD
import dill

#训练设置
set_seed(128)
args = getparse().parse_args()
device = torch.device(f'cuda:{args.GPUNUM}') if torch.cuda.is_available() else torch.device('cpu')

#加载数据
process = nerProcessor()
labellist = process.get_labels()
id2label = {i: label for i, label in enumerate(labellist)}
label2id = {label: i for i, label in enumerate(labellist)}
num_labels = len(labellist)
args.num_labels = num_labels

dev_examples = process.get_dev_examples(args.data_path)
#处理数据
config = BertConfig.from_pretrained(args.bert_path,num_labels=num_labels)
tokenizer = BertTokenizerFast.from_pretrained(args.bert_path)
dev_dataset = dizhidataset(dev_examples,tokenizer,args,labellist)
dev_loader = DataLoader(dev_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=PadBatchSeq(tokenizer.pad_token_id))
#初始化模型
model = torch.load('model.pth')
model.to(device)
loss_fun = nn.CrossEntropyLoss()
if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(dev_loader) // args.gradient_accumulation_steps) + 1
else:
    t_total = len(dev_loader) // args.gradient_accumulation_steps * 3
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ["bias", "LayerNorm.weight"]
bert_parameters = model.bert.named_parameters()
start_parameters = model.start_fc.named_parameters()
end_parameters = model.end_fc.named_parameters()
optimizer_grouped_parameters = [
    {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
     "weight_decay": args.weight_decay, 'lr': args.learning_rate},
    {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        , 'lr': args.learning_rate},

    {"params": [p for n, p in start_parameters if not any(nd in n for nd in no_decay)],
     "weight_decay": args.weight_decay, 'lr': 0.001},
    {"params": [p for n, p in start_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        , 'lr': 0.001},

    {"params": [p for n, p in end_parameters if not any(nd in n for nd in no_decay)],
     "weight_decay": args.weight_decay, 'lr': 0.001},
    {"params": [p for n, p in end_parameters if any(nd in n for nd in no_decay)], "weight_decay": 0.0
        , 'lr': 0.001},
]
args.warmup_steps = int(t_total * args.warmup_proportion)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                            num_training_steps=t_total)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
pgd = PGD(model, emb_name='word_embeddings.', epsilon=1.0, alpha=0.3)
K = 3
def train():
    tr_loss = 0
    model.zero_grad()
    pbar = ProgressBar(n_total=len(dev_loader), desc='Training', num_epochs=int(3))
    for epoch in range(3):
        pbar.reset()
        pbar.epoch_start(current_epoch=epoch)
        global_step = 0
        for step, batch in enumerate(dev_loader):
            model.train()
            token = batch['token'].long().to(device)
            attention_mask = batch['attention_mask'].long().to(device)
            start_ids = batch['start_ids'].to(device)
            end_ids = batch['end_ids'].to(device)
            start_logits, end_logits = model(token,attention_mask,start_ids)
            start_logits = start_logits.view(-1, num_labels)
            end_logits = end_logits.view(-1, num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_ids.view(-1)[active_loss]
            active_end_labels = end_ids.view(-1)[active_loss]

            start_loss = loss_fun(active_start_logits, active_start_labels)
            end_loss = loss_fun(active_end_logits, active_end_labels)
            loss = (start_loss + end_loss) / 2
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                adv_start_logits, adv_end_logits = model(token, attention_mask, start_ids)
                adv_start_logits = adv_start_logits.view(-1, num_labels)
                adv_end_logits = adv_end_logits.view(-1, num_labels)
                active_loss = attention_mask.view(-1) == 1
                adv_active_start_logits = adv_start_logits[active_loss]
                adv_active_end_logits = adv_end_logits[active_loss]

                active_start_labels = start_ids.view(-1)[active_loss]
                active_end_labels = end_ids.view(-1)[active_loss]

                adv_start_loss = loss_fun(adv_active_start_logits, active_start_labels)
                adv_end_loss = loss_fun(adv_active_end_logits, active_end_labels)
                loss_adv = (adv_start_loss + adv_end_loss) / 2
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore()
            tr_loss += loss.item()
            pbar(step, {'loss': tr_loss})

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
        torch.save(model,'new_model.pth', pickle_module=dill)

train()
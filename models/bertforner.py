import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,BertPreTrainedModel
from .layers.linears import PoolerEndLogits, PoolerStartLogits

class BertSpanForNer(BertPreTrainedModel):
    def __init__(self, config,):
        super(BertSpanForNer, self).__init__(config)
        self.soft_label = True
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, start_positions=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        return start_logits,end_logits

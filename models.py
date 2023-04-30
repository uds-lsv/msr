import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class RobertaForTokenClassification(nn.Module):

    def __init__(self, args, bert_backbone, **kwargs):
        super(RobertaForTokenClassification, self).__init__()


        self.num_labels = kwargs['num_classes']


        if bert_backbone is not None:
            assert args.load_domain_ft or args.load_af_domain_ft
            self.bert = bert_backbone
        else:
            self.bert = AutoModel.from_pretrained(args.model_name)


        self.dropout = nn.Dropout(p=args.bert_dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        attentions = outputs.attentions

        return {'logits': logits, 'last_hidden_state': sequence_output, 'attentions': attentions}



# https://curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/
class TextBert(nn.Module):

    def __init__(self, args, bert_backbone, **kwargs):
        super(TextBert, self).__init__()
        self.num_labels = kwargs['num_classes']
        assert args.pooling_strategy in ['pooler_output', 'max', 'mean']
        self.pooling_strategy = args.pooling_strategy
        self.re_init_pooler = args.re_init_pooler

        if bert_backbone is not None:
            self.bert = bert_backbone
        else:
            self.bert = AutoModel.from_pretrained(args.model_name)

        self.drop = nn.Dropout(p=args.bert_dropout_rate)
        self.out = nn.Linear(self.bert.config.hidden_size, self.num_labels)


    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = bert_out[0][:, 0, :]

        final_repr = bert_out['pooler_output']
        output = self.drop(final_repr)


        logits = self.out(output)
        return {'logits': logits, 'cls_repr': cls_repr, 'pooler_repr': final_repr}


class AdaptiveCrossEntropy(nn.Module):
    # CrossEntropy loss that works for both text classification and named-entity recognition
    def __init__(self, args, num_classes, reduction):
        super(AdaptiveCrossEntropy, self).__init__()
        self.base_ce_fn = nn.CrossEntropyLoss(reduction=reduction)


        self.num_classes = num_classes
        if args.task_type == 'ner':
            self.loss_fn = self.ner_cross_entropy
        elif args.task_type == 'text_cls':
            self.loss_fn = self.txt_cls_cross_entropy
        else:
            raise ValueError("[AdaptiveCrossEntropy]: unknown task_type")

    def txt_cls_cross_entropy(self, logits, labels, attention_mask):
        return self.base_ce_fn(logits, labels)

    def ner_cross_entropy(self, logits, labels, attention_mask):
        loss_fct = self.base_ce_fn
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_classes)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        return loss

    def forward(self, logits, labels, attention_mask=None):
        return self.loss_fn(logits, labels, attention_mask)
import copy
from typing import List
from tqdm import tqdm
import torch
from torch.utils import data
import numpy as np


class NERBertDataset(data.Dataset):
    def __init__(self, args, input_data, tokenizer, id2l):
        self.ids: List = []
        self.labels: List[List] = []
        self.examples: List = []
        self.weak_labels: List[List[List]] = []
        self.n_labels = []
        self.args = args
        self.id2l = id2l
        l2id = {v: k for k, v in id2l.items()}

        self.num_classes = len(id2l.keys())
        self.label_txt_list = [id2l[i] for i in range(self.num_classes)]
        self.tokenizer = tokenizer
        self.bert_input = None
        self.label_already_aligned = False

        if input_data is not None:  # otherwise, one needs to assign the fields manually later
            for i, item in tqdm(input_data.items()):
                self.ids.append(i)  # TODO: this id may not match the dataset index. BE CAREFUL (!)
                self.labels.append([l2id[l] for l in item['label']])
                self.weak_labels.append([[l2id[w] for w in wl] for wl in item['weak_labels']])
                self.examples.append(item['data'])

            self.weak_labels = self.ner_labels_post_processing() # get IO-formated weak_labels
            self.labels = [self.bio_id_to_io_id(np.array(l)) for l in self.labels]  # get IO-formated clean labels
            # self.num_classes = len(id2l.keys())




    def flatten_weak_labels(self):
        L = []  # weak labels,
        indexes = [0]
        for i in range(len(self)):
            L += list(self.weak_labels[i])  # size: (#tokens) * (#LF)
            indexes.append(len(self.labels[i]))
        indexes = np.cumsum(indexes)
        return np.array(L), indexes

    def bio_id_to_io_id(self, a):
        return np.where(a > 0, np.where(a % 2 == 0, a / 2, (a + 1) / 2), a).astype(int)

    def io_id_to_bio_id(self, a):
        bio_ids = []
        last_io = -1
        for i in a:
            if i == 0:
                bio_ids.append(0)
            else:
                if i == last_io:
                    bio_ids.append(int(i * 2))  # to I
                else:
                    bio_ids.append(int(i * 2 - 1))  # to B
            last_io = i
        return bio_ids


    def ner_labels_post_processing(self):
        # assert self.id2l[0] == "O", "label O should get the index zero"
        weak_labels = copy.deepcopy(self.weak_labels)

        # step 1:
        # for a token, if all LFs label "O", we do not change anything;
        # if some LFs label non-"O", we treat other LFs (which label "O") abstaining.
        for i in range(len(weak_labels)):
            for j in range(len(weak_labels[i])):
                if np.max(weak_labels[i][j]) > 0:
                    weak_labels[i][j] = [x if x > 0 else -1 for x in weak_labels[i][j]]

        self.weak_labels = weak_labels

        # step 2:
        # BIO format to IO format, i.e. convert B-XX to I-XX
        flat_weak_labels, indexes = self.flatten_weak_labels()
        flat_weak_labels = self.bio_id_to_io_id(flat_weak_labels)

        weak_labels = [np.array(flat_weak_labels[start:end]) for (start, end) in zip(indexes[:-1], indexes[1:])]

        return weak_labels

    def align_ner_labels(self, labels_to_align):

        labels = []
        for i, label in enumerate(labels_to_align):
            word_ids = self.bert_input.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                    # if data_args.label_all_tokens:
                    #     label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                    # else:
                    #     label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
            # assert len(label_ids) == len(word_ids)

        return labels

    def pad_labels(self, labels_to_pad, label_pad_token_id):
        # sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        sequence_length = self.args.max_sen_len
        padded_labels = [list(label) + [label_pad_token_id] * (sequence_length - len(label)) for label in labels_to_pad]

        return padded_labels

    def gen_bert_input(self):
        assert self.n_labels != [], "n_labels must be already fixed before generating_bert_input"
        assert self.labels != [], "labels must be already assigned before generating_bert_input"
        text_list = [i['text'] for i in self.examples]

        tokenized_inputs = self.tokenizer(
            text_list,
            padding='max_length',
            truncation=True,
            max_length=self.args.max_sen_len,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            return_tensors='pt'
        )

        self.bert_input = tokenized_inputs
        if self.label_already_aligned:
            return
        else:
            self.labels = self.align_ner_labels(labels_to_align=self.labels)
            self.n_labels = self.align_ner_labels(labels_to_align=self.n_labels)
            self.labels = torch.tensor(self.pad_labels(labels_to_pad=self.labels, label_pad_token_id=-100)).long()
            self.n_labels = torch.tensor(self.pad_labels(labels_to_pad=self.n_labels, label_pad_token_id=-100)).long()
            self.label_already_aligned = True

    def create_subset(self, idx: List[int]):
        # create a subset by index
        assert self.label_already_aligned
        assert self.n_labels != []
        dataset = NERBertDataset(self.args, None, self.tokenizer, self.id2l)

        for i in tqdm(idx, desc='creating subset'):
            dataset.ids.append(self.ids[i])
            dataset.labels.append(self.labels[i])
            dataset.examples.append(self.examples[i])
            dataset.weak_labels.append(self.weak_labels[i])
            dataset.n_labels.append(self.n_labels[i])

        dataset.label_already_aligned = True
        dataset.gen_bert_input()

        return dataset

    def get_covered_subset(self):
        assert self.label_already_aligned
        idx = [i for i in range(len(self)) if np.any(np.array(self.weak_labels[i]) != -1)]
        no_idx = [i for i in range(len(self)) if np.all(np.array(self.weak_labels[i]) == -1)]
        if len(idx) == len(self):
            return self, None
        else:
            return self.create_subset(idx), self.create_subset(no_idx)


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.bert_input.items()}

        item['c_labels'] = self.labels[index]
        # item['w_labels'] = self.weak_labels[index]
        item['n_labels'] = self.n_labels[index]
        item['index'] = index
        return item



class TextBertDataset(data.Dataset):
    def __init__(self, args, input_data, tokenizer, id2l):
        self.ids: List = []
        self.labels: List = []
        self.examples: List = []
        self.weak_labels: List[List] = []
        self.n_labels = []
        self.args = args
        self.id2l = id2l
        self.l2id = {v: k for k, v in id2l.items()}
        self.num_classes = len(id2l.keys())
        self.label_txt_list = [self.id2l[i] for i in range(self.num_classes)]
        self.tokenizer = tokenizer
        self.bert_input = None

        if input_data is not None:  # otherwise, one needs to assign the fields manually later
            for i, item in tqdm(input_data.items()):
                self.ids.append(i)  # TODO: this id may not match the dataset index. BE CAREFUL (!)
                self.labels.append(item['label'])
                self.weak_labels.append(item['weak_labels'])
                self.examples.append(item['data'])



    def gen_bert_input(self):
        text_list = [i['text'] for i in self.examples]
        self.bert_input = self.tokenizer(text_list, max_length=self.args.max_sen_len,
                                         padding="max_length", truncation=True, return_tensors='pt')

    def create_subset(self, idx: List[int]):
        # create a subset by index
        assert self.n_labels != []
        dataset = TextBertDataset(self.args, None, self.tokenizer, self.id2l)

        for i in tqdm(idx, desc='creating subset'):
            dataset.ids.append(self.ids[i])
            dataset.labels.append(self.labels[i])
            dataset.examples.append(self.examples[i])
            dataset.weak_labels.append(self.weak_labels[i])
            dataset.n_labels.append(self.n_labels[i])

        dataset.bert_input = {k: v[idx] for k, v in self.bert_input.items()}

        return dataset

    def get_covered_subset(self):
        idx = [i for i in range(len(self)) if np.any(np.array(self.weak_labels[i]) != -1)]
        no_idx = [i for i in range(len(self)) if np.all(np.array(self.weak_labels[i]) == -1)]
        if len(idx) == len(self):
            return self, None
        else:
            return self.create_subset(idx), self.create_subset(no_idx)


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.bert_input.items()}

        item['c_labels'] = self.labels[index]
        item['w_labels'] = self.weak_labels[index]
        item['n_labels'] = self.n_labels[index]
        item['index'] = index
        return item
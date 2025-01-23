from parse import args
import numpy as np
import torch
from torch import nn
import transformers
import torch
import torch.nn.functional as F
import transformers.modeling_outputs as modeling_outputs
from torch.nn import CrossEntropyLoss
from transformers import AdamW

import re
from nltk.tokenize import sent_tokenize


class OpenAI_STK(nn.Module):
    def __init__(self, DEVICE, save_path, load_path, pos_bit=1, method='OpenAI_STK'):
        super(OpenAI_STK, self).__init__()
        model = 'roberta-base-openai-detector'
        self.pos_bit = pos_bit
        self.device = DEVICE
        self.save_path = save_path
        self.load_path = load_path
        self.detector = transformers.AutoModelForSequenceClassification.from_pretrained(
            "save_models/" + model,
            num_labels=2,
            ignore_mismatched_sizes=True).to(DEVICE)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "save_models/" + model)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.detector.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in self.detector.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6)

    def forward(self, x):
        x_list = []
        with torch.no_grad():
            for i in range(len(x)):
                cur_x_split = self.split(x[i])

                encoded_input = self.tokenizer(cur_x_split, return_tensors='pt', truncation=True, padding=True).to(
                    self.device)

                sub_pred = self.detector(**encoded_input).logits.softmax(-1)
                sub_pred = sub_pred.cpu().numpy()[:, self.pos_bit]
                index = np.argsort(sub_pred)

                filter_list = []
                for ind in index:
                    if (sub_pred[ind] < args.conf_threshold):
                        filter_list.append(ind)
                filter_list = filter_list[:int(len(cur_x_split) * args.filter_threshold)]
                index = []
                for j in range(len(cur_x_split)):
                    if (j in filter_list):
                        continue
                    else:
                        index.append(j)
                # index = (index * 10)[:len(cur_x_split)]
                cur_x = ""
                for j in range(len(index)):
                    if (j == 0):
                        cur_x += cur_x_split[index[j]]
                    else:
                        if (index[j] == index[j - 1] + 1):
                            cur_x += " " + cur_x_split[index[j]]
                        else:
                            cur_x += " </s> " + cur_x_split[index[j]]
                cur_x += ""
                x_list.append(cur_x)

            encoded_input = self.tokenizer(x_list, return_tensors='pt', truncation=True, padding=True).to(
                self.device)
            outputs = self.detector(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])

        return outputs.logits.softmax(-1)

    def E_step(self, x, y):
        self.optimizer.zero_grad()

        x_list = []
        with torch.no_grad():
            for i in range(len(x)):
                cur_x_split = self.split(x[i])

                encoded_input = self.tokenizer(cur_x_split, return_tensors='pt', truncation=True, padding=True).to(
                    self.device)

                sub_pred = self.detector(**encoded_input).logits.softmax(-1)
                sub_pred = sub_pred.cpu().numpy()[:, self.pos_bit]
                index = np.argsort(sub_pred)

                filter_list = []
                for ind in index:
                    if (sub_pred[ind] < args.conf_threshold):
                        filter_list.append(ind)
                filter_list = filter_list[:int(len(cur_x_split) * args.filter_threshold)]
                index = []
                for j in range(len(cur_x_split)):
                    if (j in filter_list):
                        continue
                    else:
                        index.append(j)
                # index = (index * 10)[:len(cur_x_split)]
                cur_x = ""
                for j in range(len(index)):
                    if (j == 0):
                        cur_x += cur_x_split[index[j]]
                    else:
                        if (index[j] == index[j - 1] + 1):
                            cur_x += " " + cur_x_split[index[j]]
                        else:
                            cur_x += " </s> " + cur_x_split[index[j]]
                cur_x += ""
                x_list.append(cur_x)

        encoded_input = self.tokenizer(x_list, return_tensors='pt', truncation=True, padding=True).to(
            self.device)

        outputs = self.detector(encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'],
                                labels=y)
        self.loss = outputs[0]

    def M_step(self):
        self.loss.backward()
        self.optimizer.step()

    def save(self):
        torch.save(self.state_dict(),
                   f'%s/model_{args.conf_threshold}_{args.filter_threshold}_{args.sentence_num}_{args.iter}.pth' % self.save_path)
        # self.detector.save_pretrained(self.save_path)

    def load(self):
        self.load_state_dict(torch.load(
            f'%s/model_{args.conf_threshold}_{args.filter_threshold}_{args.sentence_num}_{args.iter}.pth' % self.load_path,
            map_location=self.device))
        # self.detector = transformers.AutoModelForSequenceClassification.from_pretrained(
        #     self.load_path,).to(self.device)

    def split(self, x):

        cur_x_split = sent_tokenize(x)
        sentence_list = cur_x_split

        sentence_list = []
        for x in cur_x_split:
            if (len(x) < 3 and len(sentence_list) > 0):
                sentence_list[-1] += ' ' + x
            else:
                sentence_list.append(x)

        if (args.sentence_num == 1):
            lens = 1
            sentence_list = sentence_list[:200]
        else:
            # lens = (len(sentence_list) + args.sentence_num - 1) // args.sentence_num
            lens = args.sentence_num
        # lens = (len(sentence_list) + 2 - 1) // 2
        nums = (len(sentence_list) - 1) // lens + 1
        paragraph_list = []
        for i in range(nums):
            begin = i * lens
            end = min(len(sentence_list), i * lens + lens)
            paragraph_list.append(" ".join(sentence_list[begin:end]))
        return paragraph_list

    def finetune(self, x, y):
        self.optimizer.zero_grad()
        encoded_input = self.tokenizer(x, return_tensors='pt', truncation=True, padding=True).to(
            self.device)
        outputs = self.detector(
            encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'], labels=y)
        loss = outputs[0]
        loss.backward()
        self.optimizer.step()

    def pretrain_forward(self, x):
        encoded_input = self.tokenizer(x, return_tensors='pt', truncation=True, padding=True).to(
            self.device)
        outputs = self.detector(
            encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])
        return outputs.logits.softmax(-1)

    def filter(self, x):
        x_list = []
        with torch.no_grad():
            for i in range(len(x)):
                cur_x_split = self.split(x[i])

                encoded_input = self.tokenizer(cur_x_split, return_tensors='pt', truncation=True, padding=True).to(
                    self.device)

                sub_pred = self.detector(**encoded_input).logits.softmax(-1)
                sub_pred = sub_pred.cpu().numpy()[:, self.pos_bit]
                index = np.argsort(sub_pred)

                filter_list = []
                for ind in index:
                    if (sub_pred[ind] < args.conf_threshold):
                        filter_list.append(ind)
                filter_list = filter_list[:int(len(cur_x_split) * args.filter_threshold)]
                index = []
                for j in range(len(cur_x_split)):
                    if (j in filter_list):
                        continue
                    else:
                        index.append(j)
                # index = (index * 10)[:len(cur_x_split)]
                cur_x = ""
                for j in range(len(index)):
                    if (j == 0):
                        cur_x += cur_x_split[index[j]]
                    else:
                        if (index[j] == index[j - 1] + 1):
                            cur_x += " " + cur_x_split[index[j]]
                        else:
                            cur_x += " </s> " + cur_x_split[index[j]]
                cur_x += ""
                x_list.append(cur_x)

        return x_list

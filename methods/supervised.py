import numpy as np
import transformers
import torch
from tqdm import tqdm
from methods.utils import timeit, cal_metrics
from torch.utils.data import DataLoader
from transformers import AdamW
import torch.utils.data as Data

from methods.ChatGPT_STK import ChatGPT_STK
from methods.ChatGPT_D import ChatGPT_D
from methods.OpenAI_STK import OpenAI_STK
from methods.MPU_STK import MPU_STK
from methods.MPU import MPU
from methods.utils import my_dataset
from parse import args
from nltk.tokenize import sent_tokenize
import re


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


@timeit
def run_supervised_experiment(
        data,
        model,
        cache_dir,
        batch_size,
        DEVICE,
        pos_bit=0,
        finetune=False,
        num_labels=2,
        epochs=5,
        save_path=None,
        load_path="",
        **kwargs):
    print(f'Beginning supervised evaluation with {model}...')
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(
        "save_models/" + model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True).to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "save_models/" + model, cache_dir=cache_dir)

    if (len(load_path) > 0):
        detector.load_state_dict(torch.load('%s/model.pth' % load_path, map_location=DEVICE))
    if finetune:
        fine_tune_model(detector, tokenizer, data, batch_size,
                        DEVICE, pos_bit, num_labels, epochs, save_path)

    train_text = data['train']['text']
    train_label = data['train']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    # detector.save_pretrained(".cache/lm-d-xxx", from_pt=True)

    if num_labels == 2:
        # train_preds = get_supervised_model_prediction(
        #     detector, tokenizer, train_text, batch_size, DEVICE, pos_bit)
        test_preds = get_supervised_model_prediction(
            detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)
    else:
        train_preds = get_supervised_model_prediction_multi_classes(
            detector, tokenizer, train_text, batch_size, DEVICE, pos_bit)
        test_preds = get_supervised_model_prediction_multi_classes(
            detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)

    predictions = {
        # 'train': train_preds,
        'test': test_preds,
    }
    # y_train_pred_prob = train_preds
    # y_train_pred = [round(_) for _ in y_train_pred_prob]
    # y_train = train_label

    y_test_pred_prob = test_preds
    y_test_pred = [round(_) for _ in y_test_pred_prob]
    y_test = test_label

    # train_res = cal_metrics(y_train, y_train_pred, y_train_pred_prob)
    test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob)
    acc_train, precision_train, tpr_train, f1_train, auc_train = 0, 0, 0, 0, 0
    acc_test, precision_test, tpr_test, f1_test, auc_test = test_res
    # print(
    #     f"{model} acc_train: {acc_train}, precision_train: {precision_train}, tpr_train: {tpr_train}, f1_train: {f1_train}, auc_train: {auc_train}")
    print(
        f"{model} acc_test: {acc_test}, precision_test: {precision_test}, tpr_test: {tpr_test}, f1_test: {f1_test}, auc_test: {auc_test}")

    # free GPU memory
    del detector
    with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()

    return {
        'name': model,
        'predictions': predictions,
        'general': {
            'acc_train': acc_train,
            'precision_train': precision_train,
            'tpr_train': tpr_train,
            'f1_train': f1_train,
            'auc_train': auc_train,
            'acc_test': acc_test,
            'precision_test': precision_test,
            'tpr_test': tpr_test,
            'f1_test': f1_test,
            'auc_test': auc_test,
        }
    }


@timeit
def run_supervised_experiment_EM(
        data,
        model,
        cache_dir,
        batch_size,
        DEVICE,
        pos_bit=0,
        finetune=False,
        num_labels=2,
        epochs=5,
        save_path=None,
        load_path="",
        **kwargs):
    print(f'Beginning supervised evaluation with {model}...')
    if (model == 'ChatGPT-STK'):
        detector = ChatGPT_STK(DEVICE, save_path, load_path, pos_bit, model)
    elif (model == 'ChatGPT-D'):
        detector = ChatGPT_D(DEVICE, save_path, load_path, pos_bit, model)
    elif (model == 'OpenAI-STK'):
        detector = OpenAI_STK(DEVICE, save_path, load_path, pos_bit, model)
    elif (model == 'MPU'):
        detector = MPU(DEVICE, save_path, load_path, pos_bit, model)
    elif (model == 'MPU-STK'):
        detector = MPU_STK(DEVICE, save_path, load_path, pos_bit, model)
    else:
        exit(0)

    if (len(load_path) > 0):
        print('load model')
        detector.load()
    if finetune:
        train_label = data['train']['label']
        if pos_bit == 0 and num_labels == 2:
            train_label = [1 if _ == 0 else 0 for _ in train_label]

        train_dataset = my_dataset(data['train']['text'], train_label)
        loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        best_acc = -1
        for i in range(epochs):
            detector.train()
            pbar = tqdm(loader, desc=f"Fine-tuning: {i} epoch")
            for (batch_x, batch_y) in pbar:
                batch_y = batch_y.to(DEVICE)
                detector.E_step(batch_x, batch_y)
                detector.M_step()
            detector.eval()
            acc_train, precision_train, recall_train, f1_train, auc_train, acc_test, precision_test, recall_test, f1_test, auc_test = eval_data(
                data, detector, batch_size, DEVICE,
                pos_bit, pre_train=False)
            if (auc_train >= best_acc):
                print('save best')
                detector.save()
                best_acc = auc_train

    detector.eval()  # 0102

    train_text = data['train']['text']
    train_label = data['train']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    if num_labels == 2:
        # train_preds = get_supervised_model_prediction_EM(
        #     detector, data['train']['text'], data['train']['label'], batch_size, DEVICE, pos_bit)
        test_preds = get_supervised_model_prediction_EM(
            detector, data['test']['text'], data['test']['label'], batch_size, DEVICE, pos_bit)
    else:
        train_preds = get_supervised_model_prediction_multi_classes(
            detector, detector.tokenizer, train_text, batch_size, DEVICE, pos_bit)
        test_preds = get_supervised_model_prediction_multi_classes(
            detector, detector.tokenizer, test_text, batch_size, DEVICE, pos_bit)

    predictions = {
        # 'train': train_preds,
        'test': test_preds,
    }
    # y_train_pred_prob = train_preds
    # y_train_pred = [round(_) for _ in y_train_pred_prob]
    # y_train = train_label

    y_test_pred_prob = test_preds
    y_test_pred = [round(_) for _ in y_test_pred_prob]
    y_test = test_label

    # train_res = cal_metrics(y_train, y_train_pred, y_train_pred_prob)
    test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob)
    acc_train, precision_train, tpr_train, f1_train, auc_train = 0, 0, 0, 0, 0
    acc_test, precision_test, tpr_test, f1_test, auc_test = test_res
    # print(
    #     f"{model} acc_train: {acc_train}, precision_train: {precision_train}, tpr_train: {tpr_train}, f1_train: {f1_train}, auc_train: {auc_train}")
    print(
        f"{model} acc_test: {acc_test}, precision_test: {precision_test}, tpr_test: {tpr_test}, f1_test: {f1_test}, auc_test: {auc_test}")

    # free GPU memory
    del detector
    with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()

    return {
        'name': model,
        'predictions': predictions,
        'general': {
            'acc_train': acc_train,
            'precision_train': precision_train,
            'tpr_train': tpr_train,
            'f1_train': f1_train,
            'auc_train': auc_train,
            'acc_test': acc_test,
            'precision_test': precision_test,
            'tpr_test': tpr_test,
            'f1_test': f1_test,
            'auc_test': auc_test,
        }
    }


def get_supervised_model_prediction(
        model,
        tokenizer,
        data,
        batch_size,
        DEVICE,
        pos_bit=0):
    with torch.no_grad():
        # get predictions for real
        preds = []
        for start in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            if (args.enhance):
                batch_data = filter(model, tokenizer, batch_data, DEVICE, pos_bit)
            else:
                batch_data = batch_data
            batch_data = tokenizer(
                batch_data,
                padding=True,
                truncation=True,
                # max_length=512,
                return_tensors="pt").to(DEVICE)
            preds.extend(model(**batch_data).logits.softmax(-1)
                         [:, pos_bit].tolist())
    return preds


def get_supervised_model_prediction_EM(
        model,
        x,
        y,
        batch_size,
        DEVICE,
        pos_bit=0,
        pre_train=False):
    train_dataset = my_dataset(x, y)
    with torch.no_grad():
        # get predictions for real
        preds = []

        loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
        pbar = tqdm(loader, "evaluation")
        for (batch_x, batch_y) in pbar:
            if (args.enhance):
                batch_data = filter(model, model.tokenizer, batch_x, DEVICE, pos_bit)
            else:
                batch_data = batch_x
            if (pre_train):
                preds.extend(model.pretrain_forward(batch_data)[:, pos_bit].tolist())
            else:
                preds.extend(model(batch_data)[:, pos_bit].tolist())

        # for start in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
        #     end = min(start + batch_size, len(data))
        #     batch_data = data[start:end]
        #     batch_data = tokenizer(
        #         batch_data,
        #         padding=True,
        #         truncation=True,
        #         max_length=512,
        #         return_tensors="pt").to(DEVICE)
        #     preds.extend(model(**batch_data).logits.softmax(-1)
        #                  [:, pos_bit].tolist())
    return preds


def get_supervised_model_prediction_multi_classes(
        model, tokenizer, data, batch_size, DEVICE, pos_bit=0):
    with torch.no_grad():
        # get predictions for real
        preds = []
        for start in tqdm(range(0, len(data), batch_size), desc="Evaluating"):
            end = min(start + batch_size, len(data))
            batch_data = data[start:end]
            batch_data = tokenizer(
                batch_data,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt").to(DEVICE)
            preds.extend(torch.argmax(
                model(**batch_data).logits, dim=1).tolist())
    return preds


def fine_tune_model(
        model,
        tokenizer,
        data,
        batch_size,
        DEVICE,
        pos_bit=1,
        num_labels=2,
        epochs=3,
        save_path=""):
    # https://huggingface.co/transformers/v3.2.0/custom_datasets.html

    train_text = data['train']['text']
    train_label = data['train']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    print(pos_bit)

    if pos_bit == 0 and num_labels == 2:
        train_label = [1 if _ == 0 else 0 for _ in train_label]
        test_label = [1 if _ == 0 else 0 for _ in test_label]

    train_encodings = tokenizer(train_text, truncation=True, padding=True)
    test_encodings = tokenizer(test_text, truncation=True, padding=True)
    train_dataset = CustomDataset(train_encodings, train_label)
    test_dataset = CustomDataset(test_encodings, test_label)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-6)

    best_acc = -1
    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Fine-tuning: {epoch} epoch"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            outputs = model(
                input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
        model.eval()
        acc_train, precision_train, recall_train, f1_train, auc_train, acc_test, precision_test, recall_test, f1_test, auc_test = eval_data(
            data, model, batch_size, DEVICE, pos_bit,
            tokenizer=tokenizer, flag=False)
        if (auc_train >= best_acc):
            torch.save(model.state_dict(), '%s/model.pth' % save_path)
            best_acc = auc_train


def eval_data(data, detector, batch_size, DEVICE, pos_bit, tokenizer=None, flag=True, pre_train=False):
    train_text = data['val']['text']
    train_label = data['val']['label']
    test_text = data['test']['text']
    test_label = data['test']['label']

    if (flag):
        train_preds = get_supervised_model_prediction_EM(
            detector, train_text, train_label, batch_size, DEVICE, pos_bit)
        test_preds = get_supervised_model_prediction_EM(
            detector, test_text, test_label, batch_size, DEVICE, pos_bit, pre_train)
    else:
        train_preds = get_supervised_model_prediction(
            detector, tokenizer, train_text, batch_size, DEVICE, pos_bit)
        test_preds = get_supervised_model_prediction(
            detector, tokenizer, test_text, batch_size, DEVICE, pos_bit)

    # predictions = {
    #     'train': train_preds,
    #     'test': test_preds,
    # }
    y_train_pred_prob = train_preds
    y_train_pred = [round(_) for _ in y_train_pred_prob]
    y_train = train_label

    y_test_pred_prob = test_preds
    y_test_pred = [round(_) for _ in y_test_pred_prob]
    y_test = test_label

    train_res = cal_metrics(y_train, y_train_pred, y_train_pred_prob)
    test_res = cal_metrics(y_test, y_test_pred, y_test_pred_prob)
    acc_train, precision_train, recall_train, f1_train, auc_train = train_res
    acc_test, precision_test, recall_test, f1_test, auc_test = test_res
    print(
        f"acc_train: {acc_train}, precision_train: {precision_train}, recall_train: {recall_train}, f1_train: {f1_train}, auc_train: {auc_train}")
    print(
        f"acc_test: {acc_test}, precision_test: {precision_test}, recall_test: {recall_test}, f1_test: {f1_test}, auc_test: {auc_test}")

    return acc_train, precision_train, recall_train, f1_train, auc_train, acc_test, precision_test, recall_test, f1_test, auc_test


def filter(model, tokenizer, x, DEVICE, pos_bit):
    x_list = []
    with torch.no_grad():
        for i in range(len(x)):
            cur_x_split = split(x[i])

            batch_data = tokenizer(
                cur_x_split,
                padding=True,
                truncation=True,
                return_tensors="pt").to(DEVICE)
            if (args.method == 'MPU'):
                sub_pred = model(cur_x_split).cpu().numpy()[:, pos_bit]
            else:
                sub_pred = model(**batch_data).logits.softmax(-1).cpu().numpy()[:, pos_bit]

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
                        # cur_x += " " + cur_x_split[index[j]]
                        cur_x += " </s> " + cur_x_split[index[j]]
            cur_x += ""
            x_list.append(cur_x)

    return x_list


def split(x):
    cur_x_split = sent_tokenize(x)
    sentence_list = cur_x_split

    # sentence_endings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!) ')
    # cur_x_split = sentence_endings.split(x)
    # # cur_x_split = [s.strip(" ") for s in cur_x_split]

    sentence_list = []
    for x in cur_x_split:
        if (len(x) < 3 and len(sentence_list) > 0):
            sentence_list[-1] += ' ' + x
        else:
            sentence_list.append(x)

    lens = args.sentence_num
    nums = (len(sentence_list) - 1) // lens + 1
    paragraph_list = []
    for i in range(nums):
        begin = i * lens
        end = min(len(sentence_list), i * lens + lens)
        paragraph_list.append(" ".join(sentence_list[begin:end]))
    return paragraph_list

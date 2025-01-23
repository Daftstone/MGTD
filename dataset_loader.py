from parse import args
import random
import datasets
import tqdm
import pandas as pd
import re

# you can add more datasets here and write your own dataset parsing function
DATASETS = ['TruthfulQA', 'SQuAD1', 'NarrativeQA', "Essay", "Reuters", "WP"]


def process_spaces(text):
    return text.replace(
        ' ,', ',').replace(
        ' .', '.').replace(
        ' ?', '?').replace(
        ' !', '!').replace(
        ' ;', ';').replace(
        ' \'', '\'').replace(
        ' ’ ', '\'').replace(
        ' :', ':').replace(
        '<newline>', '\n').replace(
        '`` ', '"').replace(
        ' \'\'', '"').replace(
        '\'\'', '"').replace(
        '.. ', '... ').replace(
        ' )', ')').replace(
        '( ', '(').replace(
        ' n\'t', 'n\'t').replace(
        ' i ', ' I ').replace(
        ' i\'', ' I\'').replace(
        '\\\'', '\'').replace(
        '\n ', '\n').strip()


def process_text_truthfulqa_adv(text):
    if "I am sorry" in text:
        first_period = text.index('.')
        start_idx = first_period + 2
        text = text[start_idx:]
    if "as an AI language model" in text or "As an AI language model" in text:
        first_period = text.index('.')
        start_idx = first_period + 2
        text = text[start_idx:]
    return text


def check_period(texts):
    for i in range(len(texts)):
        if texts[i][-1] != ".":
            texts[i] = str(texts[i]) + "."
    return texts


def check_period_single(texts):
    if texts[-1] != ".":
        texts += "."
    return texts


def load_TruthfulQA(detectLLM):
    f = pd.read_csv("datasets/TruthfulQA_LLMs.csv")
    q = f['Question'].tolist()
    a_human = f['Best Answer'].tolist()
    a_human = check_period(a_human)
    # print(a_human)
    a_chat = f[f'{detectLLM}_answer'].fillna("").tolist()
    c = f['Category'].tolist()

    res = []
    for i in range(len(q)):
        if len(
                a_human[i].split()) > 1 and len(
            a_chat[i].split()) > 1 and len(
            a_chat[i]) < 2000:
            res.append([q[i], q[i] + " " + a_human[i], q[i] + " " + a_chat[i], c[i]])
            # res.append([q[i], q[i] + " " + a_human[i], q[i] + " " + a_chat[i]])

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        },
        'val': {
            'text': [],
            'label': [],
        }

    }

    index_list = list(range(len(res)))
    # random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        elif i >= total_num * 0.8 and i < total_num * 0.9:
            data_partition = 'val'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][2]))
        data_new[data_partition]['label'].append(1)

        # data_new[data_partition]['category'].append(res[index_list[i]][3])
        # data_new[data_partition]['category'].append(res[index_list[i]][3])

    return data_new


def load_SQuAD1(detectLLM):
    f = pd.read_csv("datasets/SQuAD1_LLMs.csv")
    q = f['Question'].tolist()
    a_human = [eval(_)['text'][0] for _ in f['answers'].tolist()]
    a_chat = f[f'{detectLLM}_answer'].fillna("").tolist()

    res = []
    for i in range(len(q)):
        if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
            a_human[i] = check_period_single(a_human[i])
            res.append([q[i], q[i].rstrip(" ") + " " + a_human[i], q[i].rstrip(" ") + " " + a_chat[i]])

    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        },
        'val': {
            'text': [],
            'label': [],
        }

    }

    index_list = list(range(len(res)))
    # random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.5:
            data_partition = 'train'
        elif i >= total_num * 0.5 and i < total_num * 0.75:
            data_partition = 'val'
        else:
            data_partition = 'test'

        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][2]))
        data_new[data_partition]['label'].append(1)
    # data_new['val'] = data_new['test']
    return data_new


def load_NarrativeQA(detectLLM):
    f = pd.read_csv("datasets/NarrativeQA_LLMs.csv")
    q = f['Question'].tolist()
    a_human = f['answers'].tolist()
    a_human = [_.split(";")[0] for _ in a_human]
    a_chat = f[f'{detectLLM}_answer'].fillna("").tolist()

    res = []
    for i in range(len(q)):
        if len(
                a_human[i].split()) > 1 and len(
            a_chat[i].split()) > 1 and len(
            a_human[i].split()) < 150 and len(
            a_chat[i].split()) < 150:
            a_human[i] = check_period_single(a_human[i])
            res.append([q[i], q[i] + " " + a_human[i], q[i] + " " + a_chat[i]])


    data_new = {
        'train': {
            'text': [],
            'label': [],
        },
        'test': {
            'text': [],
            'label': [],
        },
        'val': {
            'text': [],
            'label': [],
        }

    }

    index_list = list(range(len(res)))
    # random.seed(0)
    random.shuffle(index_list)

    total_num = len(res)
    for i in tqdm.tqdm(range(total_num), desc="parsing data"):
        if i < total_num * 0.8:
            data_partition = 'train'
        elif i >= total_num * 0.8 and i < total_num * 0.9:
            data_partition = 'val'
        else:
            data_partition = 'test'
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][1]))
        data_new[data_partition]['label'].append(0)
        data_new[data_partition]['text'].append(
            process_spaces(res[index_list[i]][2]))
        data_new[data_partition]['label'].append(1)
    return data_new


def load(name, detectLLM):
    if name in ['TruthfulQA', 'SQuAD1', 'NarrativeQA']:
        load_fn = globals()[f'load_{name}']
        return load_fn(detectLLM)
    elif name in ["Essay", "Reuters", "WP"]:

        f = pd.read_csv(f"datasets/{name}_LLMs.csv")
        a_human = f["human"].tolist()
        a_chat = f[f'{detectLLM}'].fillna("").tolist()

        res = []
        for i in range(len(a_human)):
            if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
                res.append([a_human[i], a_chat[i]])

        data_new = {
            'train': {
                'text': [],
                'label': [],
            },
            'test': {
                'text': [],
                'label': [],
            },
            'val': {
                'text': [],
                'label': [],
            }

        }

        index_list = list(range(len(res)))
        # random.seed(0)
        random.shuffle(index_list)

        total_num = len(res)
        for i in tqdm.tqdm(range(total_num), desc="parsing data"):
            if i < total_num * 0.5:
                data_partition = 'train'
            elif i >= total_num * 0.5 and i < total_num * 0.75:
                data_partition = 'val'
            else:
                data_partition = 'test'
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][0]))
            data_new[data_partition]['label'].append(0)
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][1]))
            data_new[data_partition]['label'].append(1)
        return data_new
    elif name in ["CH3"]:
        from datasets import load_dataset
        all_data = load_dataset("Hello-SimpleAI/HC3", split="all")
        a_human = []
        a_chat = []
        human_answers = all_data["human_answers"]
        chatgpt_answers = all_data["chatgpt_answers"]
        for i in range(len(human_answers)):
            if (len(human_answers[i]) > 0 and len(chatgpt_answers[i]) > 0):
                a_human.append(human_answers[i][0])
                a_chat.append(chatgpt_answers[i][0])
        res = []
        for i in range(len(a_human)):
            if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
                res.append([a_human[i], a_chat[i]])

        data_new = {
            'train': {
                'text': [],
                'label': [],
            },
            'test': {
                'text': [],
                'label': [],
            },
            'val': {
                'text': [],
                'label': [],
            }

        }

        index_list = list(range(len(res)))
        # random.seed(0)
        random.shuffle(index_list)

        total_num = len(res)
        for i in tqdm.tqdm(range(total_num), desc="parsing data"):
            if i < 1000:
                data_partition = 'train'
            elif (i >= 1000 and i < 2000):
                data_partition = 'val'
            elif (i >= 2000 and i < 3000):
                data_partition = 'test'
            else:
                continue
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][0]))
            data_new[data_partition]['label'].append(0)
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][1]))
            data_new[data_partition]['label'].append(1)
        return data_new
    elif name in ["xsum"]:
        a_human = pd.read_json('datasets/xsum/en_human_lines.jsonl', lines=True)['text'].tolist()
        a_chat = pd.read_json(f'datasets/xsum/en_{detectLLM}_lines.jsonl', lines=True)['text'].tolist()

        max_num = min(len(a_human), len(a_chat))

        res = []
        for i in range(max_num):
            if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
                res.append([a_human[i], a_chat[i]])

        data_new = {
            'train': {
                'text': [],
                'label': [],
            },
            'test': {
                'text': [],
                'label': [],
            },
            'val': {
                'text': [],
                'label': [],
            }

        }

        index_list = list(range(len(res)))
        # random.seed(0)
        random.shuffle(index_list)

        total_num = len(res)
        for i in tqdm.tqdm(range(total_num), desc="parsing data"):
            if i < 1000:
                data_partition = 'train'
            elif i >= 2000 and i < 2500:
                data_partition = 'val'
            elif i >= 2500 and i < 3000:
                data_partition = 'test'
            else:
                continue
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][0]))
            data_new[data_partition]['label'].append(0)
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][1]))
            data_new[data_partition]['label'].append(1)
        return data_new
    elif name in ["goodnews"]:
        import json
        import numpy as np

        # JSON 文件路径
        file_path = 'datasets/goodnews/visualnews_test-EleutherAI_gpt-neox-20b-art1100-seg1.json'

        # 打开文件并读取内容
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        a_human = []
        a_chat = []
        for cur_data in data:
            if (np.sum(cur_data['config_dict']['mixed_labels'][:3]) > 0):
                a_human.append(" ".join(cur_data['original_sentences']))
                a_chat.append(" ".join(cur_data['merge_sentences']))
        print(len(a_human))
        max_num = len(a_human)

        res = []
        for i in range(max_num):
            if len(a_human[i].split()) > 1 and len(a_chat[i].split()) > 1:
                res.append([a_human[i], a_chat[i]])

        data_new = {
            'train': {
                'text': [],
                'label': [],
            },
            'test': {
                'text': [],
                'label': [],
            },
            'val': {
                'text': [],
                'label': [],
            }

        }

        index_list = list(range(len(res)))
        # random.seed(0)
        random.shuffle(index_list)

        total_num = len(res)
        for i in tqdm.tqdm(range(total_num), desc="parsing data"):
            if i < total_num * 0.8:
                data_partition = 'train'
            elif i >= total_num * 0.8 and i < total_num * 0.9:
                data_partition = 'val'
            else:
                data_partition = 'test'
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][0]))
            data_new[data_partition]['label'].append(0)
            data_new[data_partition]['text'].append(
                process_spaces(res[index_list[i]][1]))
            data_new[data_partition]['label'].append(1)
        return data_new
    else:
        raise ValueError(f'Unknown dataset {name}')

import json
from nltk import tokenize
import re
import spacy
import os
from collections import defaultdict

def plain(text):
    text = tokenize.sent_tokenize(text)[0]
    regx = ':.*?:`'
    text = re.sub(regx, '', text)
    regx = '\(.*\)'
    text = re.sub(regx, '', text)
    text = text.replace('`','')
    text = text.replace('~','')
    return text

def extract(text, api_name):
    res = []
    key_words = []
    doc = nlp(text)
    for token in doc:
        if token.pos_ in ['VERB','NOUN']:
            key_words.append(token.lemma_)

    pattern = r'\.|\_'
    words = re.split(pattern, api_name)
    # print('{} | {} | {}'.format(api_name, key_words, words))
    for i in key_words:
        if len(i) > 1:
            res.append(i)
    # for i in words:
    #     if len(i) > 1:
    #         res.append(i)
    print(res)
    return res

def preprocess():
    with open('data/pytorch/api_knowledge.json', 'r') as f:
        api_knowledge = json.load(f)
    postprocess = dict()

    for api_name, info in api_knowledge.items():
        api_name = api_name.replace('()','')
        s_des = info['short_description']
        if s_des:
            # print(plain(s_des))
            context_words = extract(plain(s_des), api_name)
            # postprocess[api_name.replace('()','')] = plain(s_des)
    return postprocess

def remove_tag(text):
    regx = "<(?!/?code)[^>]+>"
    text = re.sub(regx, '', text)
    text = re.sub("&#.*?;",' ', text)
    return text.strip()

def traverseList(nestList):
    flatList = []
    for item in nestList:
        if isinstance(item, list):
            flatList.extend(traverseList(item))
        else:
            flatList.append(item)
    return flatList


def find_code_by_label(text):
    pattern = "<code>(.*?)</code>"
    res = re.findall(pattern, text)
    return res

def store(file, path):
    with open(path,'w') as f:
        json.dump(file, f)

def load_defaultdict(path):
    with open(path,'r') as f:
        data = json.load(f)
    defulaut_dict = defaultdict(lambda: math.log(750000/1))
    for key in data.keys():
        defulaut_dict[key] = defulaut_dict[data[key]]
    return default_dict

def load_trn_sents(datasets):
    # with <code>
    data = []
    data_path = "/home/ec2-user/BERTOverflow/Data/release_dataset/trainset"
    for dataset in datasets:
        cur_path = os.path.join(data_path, '{}_sent.trn'.format(dataset))
        with open(cur_path, 'r') as f:
            tmp_data = json.load(f)
        data.extend(tmp_data)
    return data

#
# nlp = spacy.load("en_core_web_sm")
# preprocess()
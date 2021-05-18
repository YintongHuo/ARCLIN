import json
# from comp_idf_in_trainset import comp_idf
# from utils import traverseList, load_defaultdict
from gensim.models import FastText
from collections import Counter, defaultdict
import numpy as np
import math
from tqdm import tqdm
import os
from context_pred import ContextPredictor, BERTContext
# calculate representation
# examine test set

def load_defaultdict(path):
    with open(path,'r') as f:
        data = json.load(f)
    default_dict = defaultdict(lambda: math.log(750000/1))
    for key in data.keys():
        default_dict[key] = data[key]
    return default_dict

def traverseList(nestList):
    flatList = []
    for item in nestList:
        if isinstance(item, list):
            flatList.extend(traverseList(item))
        else:
            flatList.append(item)
    return flatList

class sent():
    def __init__(self, contx_pred):
        self.full_tokens = []
        self.pos_id = []
        self.preds = [] #len(full_token)
        self.gt = []
        self.contx_pred = contx_pred
    def append_token(self, token):
        self.full_tokens.append(token)
    def append_pos_id(self, pos_id):
        self.pos_id.append(pos_id)
    def append_preds(self, cands):
        #cands, 5 cands for api, or no
        self.preds.append(cands)
    def append_gt(self, gt):
        self.gt.append(gt)
    def get_category(self):
        sentence = ' '.join(self.full_tokens)
        self.category = self.contx_pred.predict(sentence)



def load_api(datasets):
    apis = {}
    api_path = 'API'
    for dataset in datasets:
        cur_path = os.path.join(api_path, '{}_api_knowledge.json'.format(dataset))
        with open(cur_path) as f:
            cur_knowledge = json.load(f)
        apis.update(cur_knowledge)
    print('We have {} APIs'.format(len(apis)))
    return apis


class Model():
    def __init__(self, full_datasets):
        self.apis = list(load_api(full_datasets).keys())
        self.apis = [i.replace('()', '') for i in self.apis]
        # self.entropy_token = self.entropy(self.apis)
        self.token_idf = self.TokenIDF(self.apis)

        self.class_api = {}
        for d in full_datasets:
            self.class_api[d] = list(load_api([d]).keys())
            self.class_api[d] = [i.replace('()', '') for i in self.class_api[d]]
        print('We have {} libs'.format(len(self.class_api)))

    # def entropy(self, apis):
    #     tokens = []
    #     num = len(apis)
    #     token_entropy = {}
    #     for api in apis:
    #         tokens.extend(api.split('.'))
    #     token_freq = dict(Counter(tokens))
    #     for key in token_freq.keys():
    #         token_entropy[key] = abs(-(math.log(token_freq[key]) - math.log(num)))
    #     return token_entropy

    def TokenIDF(self, apis):
        num = len(apis)
        token_idf = defaultdict(int)
        for api in apis:
            tokens = api.split('.')
            for token in tokens:
                token_idf[token] += 1
        for token in token_idf.keys():
            token_idf[token] = math.log(num/(token_idf[token]+1))
        token_idf = dict(token_idf)

        return token_idf


    def gen_cands(self, entity, categories, m_type):
        # input an entity, output the top-K similar candidates.
        cands = []
        last_entity = entity.split('.')[-1]
        if m_type:
            category = m_type
        else:
            category = categories
        # for category in categories:
            # if category:
        for api in (self.class_api[category]):
            last_api = api.split('.')[-1]
            if last_entity[:2].lower() == last_api[:2].lower():
                cands.append(api)
        return cands


    def Jaccard_sim(self, entity, api):
        queries = entity.split('.')
        answers = api.split('.')
        #tokens = list(set(queries+answers))

        salience = dict() # saliance measures the importance
        for i in answers:
            salience[i.lower()] = self.token_idf[i]

        queries = [i.lower() for i in queries]
        answers = [i.lower() for i in answers]

        # weighted_base_jaccard
        matches = [i for i in queries if i in answers]
        return sum([salience[i] for i in matches])/sum([salience[i] for i in answers])

        # return len([i for i in queries if i in answers]) / len(list(set(queries+answers)))

    def normalize(self, mention):
        # change a mention to the normalized version [in lists]
        # x.y.z: x.y, y.z, x.y.z #three type
        # x=y: x.y, s.t(if exists)
        # case_sensitive and no brackets
        mention = mention.replace('()','')
        if '(' in mention:
            mention = mention[:mention.find('(')]

        res = [mention]
        #x.y.z
        if len(mention.split('.'))>2:
            p = mention.split('.')
            res.extend('{}.{}'.format(p[i], p[i + 1]) for i in range(len(p) - 1))
        #if = exists
        res = traverseList([i.split('=') for i in res])
        return res


def idf_sim(mention, entity, idf, fasttextmodel):
    mentions = [i for i in mention.split('.') if i]
    entities = entity.split('.')
    idfs = 0
    weighted_sim = 0
    for mention in mentions:
        idfs += idf[mention]
        max_sim = -1
        for entity in entities:
            sim = fasttextmodel.wv.similarity(mention, entity)
            if sim > max_sim:
                max_sim = sim
        weighted_sim += idf[mention] * max_sim

    if idfs == 0:
        return -1
    else:
        return weighted_sim/idfs


def format_recog_output(tokens, labels, contx_pred):
    data = [] #list of sent class

    for each_tokens, each_labels in zip(tokens, labels):
        sentence = sent(contx_pred)
        for label in each_labels:
            sentence.append_pos_id(label)
        for token in each_tokens:
            sentence.append_token(token)
        sentence.get_category()
        data.append(sentence)

    print('API recog loaded...')
    print(data[0].full_tokens)
    # print(data[1].full_tokens)
    return data


def main_link(tokens, labels):
    # root = os.getcwd()
    os.chdir('/home/ec2-user/APILinking')
    fasttextmodel = FastText.load('/home/ec2-user/APILinking/fasttext/fasttext.bin')
    contx_pred = ContextPredictor()

    #pred_dataset = 'ensemble'
    full_datasets = ['pytorch','np','tf','pd','matplotlib']
    idf = load_defaultdict('pretrain/token_idf.json')
    model = Model(full_datasets)

    print('start prediction!')

    all_data = format_recog_output(tokens, labels, contx_pred)
    # for single sentence

    for data in all_data:
        for token_idx in range(len(data.pos_id)):
            if data.pos_id[token_idx] == 'B' and len(data.full_tokens[token_idx]) > 1:
                token = data.full_tokens[token_idx]
                #token_prefix = token.split('.')
                token.replace('df', 'DataFrame')
                token.replace('np', 'numpy')
                token.replace('pd', 'pandas')

                if 'DataFrame' in token:
                    m_type = 'pd'
                elif 'numpy' in token:
                    m_type = 'np'
                elif 'pandas' in token:
                    m_type = 'pd'
                elif 'tf' in token:
                    m_type = 'tf'
                else:
                    m_type = ''

                mentions = model.normalize(token)

                preds = []
                for mention in mentions:
                    cands = model.gen_cands(mention, data.category, m_type)
                    scores = []
                    for cand in cands:

                        bag_sim = idf_sim(mention, cand, idf, fasttextmodel)
                        bag_sim += idf_sim(cand, mention, idf, fasttextmodel)
                        Jaccard_sim = model.Jaccard_sim(mention, cand)
                        scores.append([cand, Jaccard_sim + (bag_sim/2)])

                scores = sorted(scores, key=lambda x:x[1], reverse=True)

                scores = ['{}'.format(i[0]) for i in scores if i[1] >= 1.1]
                if scores:

                    data.append_preds(scores[0]) #hightest one
                else:
                    data.append_preds('')
            else:
                data.append_preds('')
        # print(data.full_tokens)
        # print(data.preds)

    return [data.preds for data in all_data]

    #
    # for single_dataset in full_datasets:
    #     # precision
    #     link_pos = 0
    #     link_all = 0
    #     # recall
    #     mention_all = 0
    #     single_flag = 0
    #
    #     output_file = []
    #     for idx, data in enumerate(all_data):
    #
    #         prediction = data.preds
    #         gt = gt_data[idx]
    #         assert len(prediction) == len(gt), 'information error'
    #         mention_all += len([i for i in gt if (i in model.class_api[single_dataset])])
    #
    #         link_all += len([i for i in prediction if i])
    #         link_pos += len([i for i in range(len(prediction)) if
    #                          (prediction[i] and gt[i] == prediction[i] and (gt[i] in model.class_api[single_dataset]))])
    #
    #         # mention_all += len([i for i in gt if i])
    #         # link_all += len([i for i in prediction if i])
    #         # link_pos += len([i for i in range(len(prediction)) if (prediction[i] and gt[i] == prediction[i])])
    #         #print(data.full_tokens, data.preds)
    #         for token, preds, label in zip(data.full_tokens, data.preds, gt):
    #             line = ''
    #             line += '{},{},{},{},{}'.format(token, preds, label, preds==label, data.category)
    #             # if len(preds) >= 1:
    #             #     line += ' '.join(preds)
    #             line += '\n'
    #             output_file.append(line)
    #         output_file.append('\n')
    #
    #
    #
    #     f = open('prediction/tst_{}-w2v.conll'.format(single_dataset), "w")
    #     for line in output_file:
    #         f.write(line)
    #
    #     print('===RESULT===')
    #     precision = link_pos/link_all
    #     recall = link_pos/mention_all
    #     f_score = 2*precision*recall/(precision+recall)
    #     # print('Full datasets:', full_datasets)
    #     print(single_dataset)
    #     print('LinkAll:',link_all)
    #     print('LinkPos:', link_pos)
    #     print('Precision:', precision)
    #     print('Recall:', recall)
    #     print('F1 Score:', f_score)

    # threshold_combination = str(crf_threshold) + '$$' + str(score_threshold)
    # threshold_change[threshold_combination] = dict()
    # threshold_change[threshold_combination]['precision'] = precision
    # threshold_change[threshold_combination]['recall'] = recall
    # threshold_change[threshold_combination]['f1'] = f_score
    #
    # with open('threshold_change.json','w') as f:
    #     json.dump(threshold_change, f)



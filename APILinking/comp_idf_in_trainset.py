import json, sys
from tqdm import tqdm
sys.path.append('/home/ec2-user/BERTOverflow/model')
from embedding.twokenize import tokenize
from collections import defaultdict
import math
from utils import load_trn_sents

def comp_idf(datasets):
    print('Start compute idf...')

    plain_texts = load_trn_sents(datasets)
    # with open('data/pytorch/trn_sent.json', 'r') as f:
    #     plain_texts = json.load(f)
    # with open('data/pytorch/tst_sent.json', 'r') as f:
    #     test_texts = json.load(f)

    tokens_freq = defaultdict(int)
    for sent in tqdm(plain_texts):
        sent = sent.replace('<code>', '').replace('</code>', '')
        dist_tokens = list(set(tokenize(sent)))
        for i in dist_tokens:
            tokens_freq[i] += 1
    tokens_freq = dict(tokens_freq)

    sent_num = len(plain_texts)
    idf_dict = defaultdict(lambda: math.log(sent_num/1))
    for key, value in tokens_freq.items():
        idf_dict[key] = math.log(sent_num/(value+1))

    with open('pretrain/token_idf.json','w') as f:
        json.dump(idf_dict, f)

    return idf_dict


# comp_idf(['pytorch','np','tf','pd','matplotlib'])

# # total 1912 oov
# with open('token_idf.json','w') as f:
#     json.dump(idf_dict, f)

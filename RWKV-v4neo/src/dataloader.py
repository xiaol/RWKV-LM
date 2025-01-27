import numpy as np
from tqdm import tqdm
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
import sys
import json
import random
import os
from pytorch_lightning.utilities import rank_zero_info

tokenizer = TRIE_TOKENIZER('/home/neromous/pyblackfog/rwkv_trainer/rwkv_vocab_v20230424.txt')

# config = {"me":              [65530],    # robot
#           "read-system-doc": [65531],    # robot 阅读系统文档
#           "read-user-req":   [65532],    # robot 阅读user请求
#           "think":           [65533],    # 思考
#           "response":        [65534],    # 从屏幕输出结果 
#           "over":            [65535]     # 动作结束
#           }


# robot   = config['me']
# system  = config['read-system-doc']
# user    = config['read-user-req']
# think   = config['think']
# response = config['response']
# over     = config['over']

# 人员 角色 组织机构
# 增 删 改 查

config = {"<|system|>":              [65525],    # robot
          #"<|/master|>":            [65526],    # robot 阅读系统文档
          #"<|world|>":              [65527],    # robot
          #"<|/world|>":             [65528],    # robot 阅读系统文档
          "<|master|>":              [65529],    # robot 阅读系统文档
          #"<|/robot|>":             [65530],    # robot 阅读系统文档
          #"<|request|>":            [65531],     # 动作结束
          #"<|response|>":           [65532],     # 动作结束
          "<|robot|>":               [65533],     # 动作结束
          #"<|function-res|>":       [65534],     # 动作结束
          "<|over|>":                [65535],     # 动作结束
          }


def to_im_item(item):
    role = item['role']
    role = role.strip()
    # action = config[action]
    content = item['content']
    content = content.strip()
    text = f'{content}'
    output = tokenizer.encode(text)
    if role == "system":
        output = [65525] + output + [65535]
    elif role == "master":
        output = [65529] + output + [65535]
    else:
        output = [65533] + output + [65535]
    #print("tesxt====",output)
    return output

def question_answer(item):
    system = item['system']
    question = item['question']
    answer  = item['answer']
    system = to_im_item({"role"   :"system",  "content": system})
    question = to_im_item({"role" :"master", "content": question})
    answer = to_im_item({"role"   :"robot",  "content": answer})
    output =  system + question + answer
    output = np.array(output, dtype='uint16')
    return output


def logs(item):
    output = []
    for x in item:
        if x['role'] == 'system':
            r = tokenizer.encode(x['text'])
            r = [65525] + r + [65535]
            output += r
        elif x['role'] == 'master':
            r = tokenizer.encode(x['text'])
            r = [65529] + r + [65535]
            output += r
        elif x['role'] == 'robot':
            r = tokenizer.encode(x['text'])
            r = [65533] + r + [65535]
            output += r
        elif x['role'] == 'other':
            r = tokenizer.encode(x['text'])
            r = [65527] + r + [65535]
            output += r
        else:
            raise Exception
    output = np.array(output, dtype='uint16')
    return output


def text_from_json(item):
    text = item.get('text')
    text = tokenizer.encode(text)
    output = text
    output = np.array(output, dtype='uint16')
    return output

def dispatch(item):
    if type(item) == dict:
        if "text" in item.keys():
            res = text_from_json(item)
        elif "system" in item.keys() and "question" in item.keys() and "answer" in item.keys() :
            res = question_answer(item)
        else:
            raise Exception
    elif type(item) == list:
        res = logs(item)
    else:
        raise Exception
    return res

def read_jsonl(path, n=0):
    res = []
    with open(path,'r',encoding='utf-8') as f:
        data = f.readlines()
    data = [x.strip() for x in data if x.strip() != ""]
    if len(data) > n and n != 0:
        data = random.sample(data, n)
    for text in data:
        res.append(json.loads(text))
    return res


def load_file_path(path):
    root = path
    r = {'book' :      200,
         'cc'   :      2000,
         'wikipedia':  2000,
         'arxiv' :     1000,
         'orca' :      80000,
         'github' :    8000,
         'trpg' :      8000,
         'tool':       1000}
    n = r.get(path.split('/')[-1], 40000)
    if os.path.isdir(path):
        files = os.listdir(root)
        files = [x for x in files if x.endswith('.jsonl')]
        path = random.choice(files)
        data = read_jsonl(os.path.join(root,path), n)
        return data
    elif os.path.isfile(path):
        data = read_jsonl(os.path.join(root,path))
        return data
    else:
        raise Error

def dataloader(root_path):
    files = os.listdir(root_path)
    data = []
    res =  []
    for f in tqdm(files):
        data += load_file_path(os.path.join(root_path,f))
    rank_zero_info(f"total {files} files  has {len(data)} items.")
    for v in tqdm(data):
        res.append(dispatch(v))
    res = [x for x in res if len(x) != 0]
    return res

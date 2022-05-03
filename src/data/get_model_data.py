# -*- coding: utf-8 -*-
# @Time : 2022/4/30 22:24
# @Author : Jclian91
# @File : get_model_data.py
# @Place : Minghang, Shanghai
import json
import pandas as pd
from random import shuffle
from collections import defaultdict


# 将mention周围用#、#表示出来
def text_replace(text, mention):
    start_index = text.index(mention)
    end_index = start_index + len(mention)
    if end_index <= len(text):
        string = text[:start_index] + '<e>' + mention + '</e>' + text[end_index:]
    else:
        string = text[:start_index] + '<e>' + mention + '</e>'
    return string


item_dict = defaultdict(dict)
item_set = defaultdict(set)
item_text_list = defaultdict(list)
df = pd.read_excel("标注语料_20220430.xlsx")
for index in range(df.shape[0]):
    text, mention, item, url = df.iloc[index, :]
    item = f'{mention}（{item}）'
    item_dict[mention][item] = url
    item_set[mention].add(item)
    item_text_list[f"{mention}_{item}"].append(text)

print('实体数量: {}'.format(len(df['mention'].unique())))
print('义项数量: {}'.format(len(df['正确义项'].unique())))

with open("item.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(item_dict, ensure_ascii=False, indent=2))
print(item_set)
print(item_text_list)


# 划分成训练集和测试集
train_ration = 0.8
train_data_list = []
test_data_list = []

train_id = 1
test_id = 1
for key, text_list in item_text_list.items():
    mention, item = key.split('_', maxsplit=1)
    shuffle(text_list)
    train_length = int(train_ration * len(text_list))
    train_length = train_length if train_length else 1
    train_list = text_list[:train_length]
    test_list = text_list[train_length:]
    for text in train_list:
        print(mention, text)
        context = '</ec>'.join(item_set[mention]) + '</ec>'
        sample = [{
                    "context": context,
                    "qas": [
                        {
                            "answers": [
                                {
                                    "answer_start": context.index(item),
                                    "text": item
                                }
                            ],
                            "question": text_replace(text, mention),
                            "id": f"{train_id}"
                        }
                    ]
                }]
        train_data_list.append({'title': '', 'paragraphs': sample})
        train_id += 1

    for text in test_list:
        context = '</ec>'.join(item_set[mention]) + '</ec>'
        sample = [{
                    "context": context,
                    "qas": [
                        {
                            "answers": [
                                {
                                    "answer_start": context.index(item),
                                    "text": item
                                }
                            ],
                            "question": text_replace(text, mention),
                            "id": f"{test_id}"
                        }
                    ]
                }]
        test_data_list.append({'title': '', 'paragraphs': sample})
        test_id += 1

# 保存成文件
with open('ed_train.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps({'data': train_data_list}, ensure_ascii=False, indent=4))

with open('ed_test.json', 'w', encoding='utf-8') as g:
    g.write(json.dumps({'data': test_data_list}, ensure_ascii=False, indent=4))

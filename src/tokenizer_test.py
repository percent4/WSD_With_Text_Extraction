# -*- coding: utf-8 -*-
# @Time : 2022/4/30 23:21
# @Author : Jclian91
# @File : tokenizer_test.py
# @Place : Minghang, Shanghai
from tokenizers import BertWordPieceTokenizer
tokenizer = BertWordPieceTokenizer("chinese-RoBERTa-wwm-ext/vocab.txt", lowercase=True)
tokenizer.add_special_tokens(['<e>', '</e>', '</ec>'])

context = '<e>苹果</e>树尽早疏蕾，能节省营养，利于坐大果，促果高桩。'
tokenized_context = tokenizer.encode(context)
print(tokenized_context.ids)
print(len(tokenized_context.ids))
print([tokenizer.id_to_token(_) for _ in tokenized_context.ids])
print(tokenizer.get_vocab_size())

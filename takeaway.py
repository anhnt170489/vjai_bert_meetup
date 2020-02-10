from transformers.tokenization_bert import BertTokenizer

# Input
seqs = [['Donald John Trump (born June 14, 1946) is the 45th and current president of the United States',
         'Xi Jinping is a Chinese politician serving as the general secretary of the Communist Party of China (CPC)'],
        [
            'Vladimir Vladimirovich Putin is the president of Russia since 2012, previously holding the position from 2000 until 2008',
            'Kim Jong-un is a North Korean politician who is the supreme leader of North Korea since 2011 and chairman of the Workers Party of Korea since 2012']]

text = seqs[0][0]
pair = seqs[0][1]

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

tokens = tokenizer.tokenize(text)
tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
# print(tokens)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
# print(input_ids)
inputs = [tokenizer.encode_plus(
    text,
    pair,
    add_special_tokens=True,
    max_length=128,
)]
# print(inputs)

from transformers.modeling_bert import BertModel
import torch

# Load model to train from scratch
# from transformers.configuration_bert import BertConfig
# from transformers.configuration_utils import CONFIG_NAME
# import os
#
# model_path = 'models/bert_base_cased/'
# config = BertConfig.from_json_file(
#     os.path.join(model_path, CONFIG_NAME)
# )
# model = BertModel(config)

# Load pretrained model
model = BertModel.from_pretrained('bert-base-cased')

# Padding
all_input_ids = []
all_segment_ids = []
all_attention_mask = []
pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
for input in inputs:
    length_to_pad = 256 - len(input['input_ids'])
    all_input_ids.append(input['input_ids'] + ([pad_token] * length_to_pad))
    all_attention_mask.append(input['attention_mask'] + ([0] * length_to_pad))
    all_segment_ids.append(input['token_type_ids'] + ([0] * length_to_pad))

all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)

output = model(input_ids=all_input_ids, attention_mask=all_attention_mask, token_type_ids=all_segment_ids)

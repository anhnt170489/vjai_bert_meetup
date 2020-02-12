# vjai_bert_meetup
stuff related to my talk at VJAI meetup about the Transformers

# Slides:
[Main talk](https://github.com/anhnt170489/vjai_bert_meetup/blob/master/Transformers%40VJAI.pdf)
[Distillation](https://github.com/anhnt170489/vjai_bert_meetup/blob/master/Distilling%20Bert%40AIST.pdf)

# Takeaway code
## All together:
[takeaway.py](https://github.com/anhnt170489/vjai_bert_meetup/blob/master/takeaway.py)

## Use sentencepiece to build vocab
[build_vocab.py](https://github.com/anhnt170489/vjai_bert_meetup/blob/master/build_vocab.py)

## Use tokenizer
```
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

tokens = tokenizer.tokenize(text)
tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
input_ids = tokenizer.convert_tokens_to_ids(tokens)
inputs = [tokenizer.encode_plus(
    text,
    pair,
    add_special_tokens=True,
    max_length=128,
)]
```

## Use Nvidia apex amp (fp16)
```
try:
    from apex import amp
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
```

## Call BertModel
```
from transformers.modeling_bert import BertModel
import torch

# Load model to train from scratch
from transformers.configuration_bert import BertConfig
from transformers.configuration_utils import CONFIG_NAME
import os
model_path = 'models/bert_base_cased/'
config = BertConfig.from_json_file(
    os.path.join(model_path, CONFIG_NAME)
)
model = BertModel(config)

# Load pretrained model
model = BertModel.from_pretrained('bert-base-cased')
```

## Get Bert output to further use
```
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
```

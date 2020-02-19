import torch
from transformers import BertModel, BertTokenizer

if __name__ == "__main__":
    bert_path = './resources/chinese_wwm_L-12_H-768_A-12'
    # path = '/home1/lsy2018/NL2SQL/python3/bert/chinese_wwm_L-12_H-768_A-12/'
    # 加载bert的分词器
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    text = "训好的模型用这样就好了"
    tokenized_text = tokenizer.encode(text, add_special_tokens=True)
    print(tokenized_text)

    # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # print(indexed_tokens)

    # segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    tokens_tensor = torch.tensor([tokenized_text])
    # segments_tensors = torch.tensor([segments_ids])

    model = BertModel.from_pretrained(bert_path)
    with torch.no_grad():
        last_hidden_states = model(tokens_tensor)[0]

    tokens_tensor = tokens_tensor.cuda()
    segments_tensors = segments_tensors.cuda()
    model.cuda()

    # 得到每一层的 hidden states
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    # assert len(encoded_layers) == 12
    print(encoded_layers[-1].shape)
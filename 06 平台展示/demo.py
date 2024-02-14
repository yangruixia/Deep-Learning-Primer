import transformers
from transformers import BertTokenizer, BertModel
import torch

from model import myModel
import json
from tqdm import tqdm
import unicodedata, re
from data_preprocessing import tools

import argparse

parser = argparse.ArgumentParser(description="train")
parser.add_argument("--train_path", type=str, default="/home/yangruixia/workspace/sem-explore/test/input/paper_train",
                    help="train file")
parser.add_argument("--test_path", type=str, default="/home/yangruixia/workspace/sem-explore/test/input/paper_test", help="test file")
parser.add_argument("--schema_path", type=str, default="./event_schema/event_schema.json", help="schema")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/multilabel_cls.pth", help="output_dir")
parser.add_argument("--bert_mrc_checkpoints", type=str, default="./checkpoints/bert_mrc.pth", help="output_dir")
parser.add_argument("--vocab_file", type=str, default="./data/vocab.txt", help="vocab_file")
parser.add_argument("--tag_file", type=str, default="./data/tags.txt", help="tag_file")
parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
parser.add_argument("--hidden_num", type=int, default=512, help="hidden_num")
parser.add_argument("--max_length", type=int, default=128, help="max_length")
parser.add_argument("--embedding_file", type=str, default=None, help="embedding_file")
parser.add_argument("--epoch", type=int, default=400, help="epoch")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning_rate")
parser.add_argument("--require_improvement", type=int, default=100, help="require_improvement")
parser.add_argument("--pretrained_model_path", type=str, default="/home/yangruixia/workspace/sem-explore/test/pretrained_model/chinese_roberta_wwm_ext",
                    help="pretrained_model_path")
parser.add_argument("--clip_norm", type=str, default=0.25, help="clip_norm")
parser.add_argument("--warm_up_epoch", type=str, default=1, help="warm_up_steps")
parser.add_argument("--decay_epoch", type=str, default=80, help="decay_steps")
parser.add_argument("--output", type=str, default="./output/paper_result.json", help="output")

args = parser.parse_args()
from flask import Flask, request, render_template_string
# 创建 Flask 应用
app = Flask(__name__)

# 基本的 HTML 模板
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>NLP Model Prediction</title>
</head>
<body>
    <h1>输入句子</h1>
    <form method="post" action="/predict">
    <textarea name="sentence" rows="3" cols="40" placeholder="请输入句子..."></textarea><br>
    <input type="submit" value="提交">
    </form>
    {% if prediction %}
    <h2>预测结果:</h2>
    <p>{{ prediction }}</p>
    {% endif %}
</body>
</html>
"""

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

added_token = ['[unused1]', '[unused1]']
tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_path, additional_special_tokens=added_token)
label2id, id2label, num_labels = tools.load_schema()


def load_data(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentences = []
        arguments = []
        for line in lines:
            data = json.loads(line)
            text = data['text']
            entity_list = data['entity_list']
            args_dict = {}
            if entity_list != []:
                for entity in entity_list:
                    entity_type = entity['type']
                    entity_argument = entity['argument']
                    args_dict[entity_type] = entity_argument
                sentences.append(text)
                arguments.append(args_dict)
        return sentences, arguments


def get_mapping(text):
    text_token = tokenizer.tokenize(text)
    text_mapping = tools.token_rematch().rematch(text, text_token)
    return text_mapping


def sapn_decode(span_logits):
    arg_index = []
    for i in range(len(span_logits)):
        for j in range(i, len(span_logits[i])):
            if span_logits[i][j] > 0:
                arg_index.append((i, j, id2label[span_logits[i][j] - 1]))
    return arg_index




with torch.no_grad():
    model = myModel(pre_train_dir=args.pretrained_model_path, dropout_rate=0.5).to(device)
    model.load_state_dict(
        torch.load('D:\PyTorch_BERT_Biaffine_NER\checkpoints\multilabel_cls.pth', map_location=torch.device('cpu')),
        strict=False)
    sentences, _ = load_data(args.test_path)
def auto_answer(sent):
            encode_dict = tokenizer.encode_plus(sent, truncation=True, max_length=args.max_length, padding='longest')
            # encode_dict = tokenizer.encode_plus(sent,
            #                        max_length=args.max_length,
            #                        pad_to_max_length=True)
            input_ids = encode_dict['input_ids']
            input_seg = encode_dict['token_type_ids']
            input_mask = encode_dict['attention_mask']

            input_ids = torch.Tensor([input_ids]).long()
            input_seg = torch.Tensor([input_seg]).long()
            input_mask = torch.Tensor([input_mask]).float()
            span_logits = model(
                input_ids=input_ids.to(device),
                input_mask=input_mask.to(device),
                input_seg=input_seg.to(device),
                is_training=False)

            span_logits = torch.argmax(span_logits, dim=-1)[0].to(torch.device('cpu')).numpy().tolist()
            args_index = sapn_decode(span_logits)
            text_mapping = get_mapping(sent)
            entity_list = []

            for k in args_index:
                try:
                    dv = 0
                    while text_mapping[k[0] - 1 + dv] == []:
                        dv += 1
                    start_split = text_mapping[k[0] - 1 + dv]

                    while text_mapping[k[1] - 1 + dv] == []:
                        dv += 1

                    end_split = text_mapping[k[1] - 1 + dv]

                    argument = sent[start_split[0]:end_split[-1] + 1]
                    entity_type = k[2]
                    entity_list.append({'type': entity_type, 'argument': argument})
                except:
                    pass
            result = {'text': sent, 'entity_list': entity_list}
            print(result['text'])
            answer = ''
            for entity in result['entity_list']:
                if entity['type'] == 'P':
                    answer += f"P: {entity['argument']}\n"
                elif entity['type'] == 'Q':
                    answer += f"Q: {entity['argument']}\n"
                elif entity['type'] == 'R':
                    answer += f"R: {entity['argument']}\n"
                elif entity['type'] == 'NQ':
                    answer += f"NQ: {entity['argument']}\n"
            if answer == '':
                answer = '这个句子没有识别出什么东西'
            return answer


# 路由到主页面
@app.route("/", methods=["GET"])
def index():
    # 显示输入表单
    return render_template_string(HTML_TEMPLATE)

# 预测路由
@app.route("/predict", methods=["POST"])
def predict():
    sentence = request.form["sentence"]
    # 在此处添加模型预测逻辑
    # 这里简单地返回输入的句子作为"预测"结果
    prediction = auto_answer(sentence)  # 应替换为真正的预测逻辑
    # 显示结果和输入表单
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

# 主程序入口
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port='8001')

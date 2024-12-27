from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import random

app = Flask(__name__)

# 加载模型和pipeline
nlp_pipeline = pipeline("token-classification",
                        model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')


@app.route('/')
@app.route('/ner')
def ner():
    return render_template('NER.html', title="NER - Medical Data Anonymization")

@app.route('/deidentification')
def deidentification():
    return render_template('De-identification.html', title="De-identification - Medical Data Anonymization")

@app.route('/about')
def about():
    return render_template('About.html', title="About - Medical Data Anonymization")


@app.route('/process', methods=['POST'])
def process():
    text = request.form['text']
    ner_results = nlp_pipeline(text)

    # 用于处理重叠的标记
    segments = []
    current_pos = 0

    # 记录每个标签的颜色
    label_colors = {}
    COLOR_PALETTE = [
        "#e0c3fc", "#bde0fe", "#ffccf9", "#ffd6a5", "#caffbf", "#9bf6ff",
        "#ffc6ff", "#fdffb6", "#c3aed6", "#a0c4ff", "#bdb2ff", "#ffafcc"
    ]

    def assign_color(label):
        if label not in label_colors:
            if len(label_colors) < len(COLOR_PALETTE):
                label_colors[label] = COLOR_PALETTE[len(label_colors)]
            else:
                label_colors[label] = f'#{random.randint(0, 0xFFFFFF):06x}'
        return label_colors[label]

    ner_results = sorted(ner_results, key=lambda x: x['start'])

    for entity in ner_results:
        start, end = entity['start'], entity['end']
        label = entity['entity_group']
        label_color = assign_color(label)

        # 添加当前未被标记的文本部分
        if current_pos < start:
            segments.append({'text': text[current_pos:start], 'label': None})

        # 添加当前标记的文本部分
        segments.append({'text': text[start:end], 'label': label, 'color': label_color})

        # 更新当前位置
        current_pos = end

    # 添加最后未被标记的部分
    if current_pos < len(text):
        segments.append({'text': text[current_pos:], 'label': None})

    # 根据段落生成HTML
    result_str = ""
    for segment in segments:
        if segment['label']:
            result_str += f'<span style="background-color:{segment["color"]}; ' \
                          f'padding: 5px 10px; margin: 2px; border-radius: 25px; ' \
                          f'box-shadow: 0px 1px 5px rgba(0, 0, 0, 0.2); font-weight: bold;">'
            result_str += f'{segment["text"]} <span style="background-color:rgba(0, 0, 0, 0.1); ' \
                          f'color: #fff; padding: 2px 4px; border-radius: 5px; font-size: 0.8em;">' \
                          f'{segment["label"]}</span>'
            result_str += '</span>'
        else:
            result_str += segment['text']

    return jsonify({'result': result_str})


@app.route('/process_deid', methods=['POST'])
def process_deid():
    text = request.form['text']
    ner_results = nlp_pipeline(text)

    # 将识别到的实体信息整理成列表, 并同时返回给前端，便于选择策略
    # 格式示例：[{entity_group: "AGE", start:10, end:13, text:"17-year-old"}, ...]
    entities = []
    for entity in ner_results:
        entities.append({
            'entity_group': entity['entity_group'],
            'start': entity['start'],
            'end': entity['end'],
            'text': text[entity['start']:entity['end']]
        })

    # 按entity_group分类
    entity_groups = list(set([e['entity_group'] for e in entities]))

    return jsonify({'entities': entities, 'entity_groups': entity_groups, 'original_text': text})


@app.route('/apply_deid', methods=['POST'])
def apply_deid():
    # 前端会传递:
    # original_text: 原文
    # strategies: {entity_group: selected_strategy}
    # 例如: {"AGE": "delete", "DATE": "generalize", ...}
    data = request.get_json()
    original_text = data['original_text']
    strategies = data['strategies']
    entities = data['entities']

    # 根据策略对原文进行替换，这里简单示例
    # 假设：
    # - delete策略会将对应文本替换为"[REDACTED]"
    # - generalize策略这里简单做一个示例（比如用"X"替代字符）
    # - pseudonymize策略以"[PSEUDONYM]"替代
    # 实际使用时应根据具体需求实现相应逻辑。
    replaced_text = original_text
    # 为了避免多次替换时的索引问题，我们从后向前替换
    for ent in sorted(entities, key=lambda x: x['start'], reverse=True):
        eg = ent['entity_group']
        if eg in strategies:
            strategy = strategies[eg]
            if strategy == "delete":
                replaced_text = replaced_text[:ent['start']] + "[REDACTED]" + replaced_text[ent['end']:]
            elif strategy == "generalize":
                # 简单泛化，这里全部替换为相同长度的‘X’
                length = ent['end'] - ent['start']
                replaced_text = replaced_text[:ent['start']] + "X"*length + replaced_text[ent['end']:]
            elif strategy == "pseudonymize":
                replaced_text = replaced_text[:ent['start']] + "[PSEUDONYM]" + replaced_text[ent['end']:]

    return jsonify({'deidentified_text': replaced_text})


if __name__ == '__main__':
    app.run(debug=True)

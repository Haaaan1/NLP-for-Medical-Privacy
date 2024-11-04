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


if __name__ == '__main__':
    app.run(debug=True)

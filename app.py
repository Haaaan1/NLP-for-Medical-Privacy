from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import random
import re
import json
from datetime import datetime
import dateparser
from dateutil import parser
import pandas as pd
import io

app = Flask(__name__)

# Load model and pipeline
nlp_pipeline = pipeline(
    "token-classification",
    model="Clinical-AI-Apollo/Medical-NER",
    aggregation_strategy='simple'
)

nlp_pipeline_germain = pipeline(
    "token-classification",
    model="HUMADEX/german_medical_ner",
    aggregation_strategy='simple'
)

# Predefined label colors for known entities
LABEL_COLORS = {
    "AGE": "#e0c3fc",
    "HISTORY": "#bde0fe",
    "SEX": "#ffccf9",
    "CLINICAL_EVENT": "#ffd6a5",
    "DURATION": "#caffbf",
    "BIOLOGICAL_STRUCTURE": "#9bf6ff",
    "SIGN_SYMPTOM": "#ffc6ff",
    "DISEASE_DISORDER": "#fdffb6",
    "DATE": "#c3aed6",
    "LAB_VALUE": "#a0c4ff",
    "DIAGNOSTIC_PROCEDURE": "#bdb2ff",
    "DETAILED_DESCRIPTION": "#ffafcc",
    "THERAPEUTIC_PROCEDURE": "#ff9e9d",
    "DOSAGE": "#c3ffd8",
    "ADMINISTRATION": "#a9d6e5",
    "MEDICATION": "#dab6fc"
}

# Fallback color palette for dynamically assigned colors
COLOR_PALETTE = [
    "#e0c3fc", "#bde0fe", "#ffccf9", "#ffd6a5", "#caffbf", "#9bf6ff",
    "#ffc6ff", "#fdffb6", "#c3aed6", "#a0c4ff", "#bdb2ff", "#ffafcc"
]


# Assign color for new labels if they are not in LABEL_COLORS
def assign_color(label):
    if label not in LABEL_COLORS:
        if len(LABEL_COLORS) < len(COLOR_PALETTE):
            LABEL_COLORS[label] = COLOR_PALETTE[len(LABEL_COLORS) % len(COLOR_PALETTE)]
        else:
            LABEL_COLORS[label] = f'#{random.randint(0, 0xFFFFFF):06x}'
    return LABEL_COLORS[label]


def get_confidence_color(confidence):
    if confidence >= 0.8:
        return "green"
    elif 0.6 <= confidence < 0.8:
        return "yellow"
    else:
        return "red"


# Define entity weights
ENTITY_WEIGHTS = {
    "AGE": 2,
    "SEX": 1,
    "DATE": 2,
    "DURATION": 1,
    "HISTORY": 2,
    "DETAILED_DESCRIPTION": 2,
    "DIAGNOSTIC_PROCEDURE": 1,
    "CLINICAL_EVENT": 1,
    "DISEASE_DISORDER": 2,
    "SIGN_SYMPTOM": 2,
    "THERAPEUTIC_PROCEDURE": 1,
    "MEDICATION": 2,
}


def calculate_privacy_score(entities):
    total_entities = len(entities)
    if total_entities == 0:
        return 0.0

    privacy_entity_count = sum(
        1 for e in entities if e['entity_group'] in ENTITY_WEIGHTS
    )
    quantitative_score = privacy_entity_count / total_entities

    appeared_types = set(
        e['entity_group'] for e in entities if e['entity_group'] in ENTITY_WEIGHTS
    )
    total_weight = sum(ENTITY_WEIGHTS.values())
    normalized_entity_weights = {key: value / total_weight for key, value in ENTITY_WEIGHTS.items()}

    qualitative_score = sum(normalized_entity_weights[t] for t in appeared_types)

    alpha = 0.5
    final_score = alpha * quantitative_score + (1 - alpha) * qualitative_score
    return final_score


def generalize_age(age_str, level="mild", language="en"):
    # 使用传入的 language 参数，不再自动检测输入字符串中的语言
    pattern = re.compile(
        r'\b(\d+)[\s-]*(?:year-old|years old|years-old|yr-old|years|year|yrs|'
        r'Jahre(?:[\s-]*(?:alt))?|Jahr(?:[\s-]*(?:alt))?|jährig|jährige)\b'
        r'|\b(?:age|alter)[\s-]*[:\-]?\s*(\d+)\b',
        re.IGNORECASE
    )

    def replace_age(match):
        num_str = match.group(1) if match.group(1) else match.group(2)
        if not num_str:
            return "[AGE]"
        try:
            original_age = int(num_str)
        except ValueError:
            return "[AGE]"

        if level == "mild":
            interval = 5
        elif level == "moderate":
            interval = 10
        elif level == "severe":
            interval = 20
        else:
            interval = 5

        lower_bound = (original_age // interval) * interval
        upper_bound = lower_bound + interval

        # 根据传入的 language 参数添加后缀
        if language == "de":
            return f"{lower_bound}-{upper_bound} Jahre alt"
        else:
            return f"{lower_bound}-{upper_bound} years old"

    generalized_str = pattern.sub(replace_age, age_str)

    if generalized_str == age_str:
        return "[ALTER]" if language == "de" else "[AGE]"
    return generalized_str


def generalize_date(date_str, level="mild", language="en"):
    # 使用 dateparser 解析日期，并指定解析语言
    parsed_date = dateparser.parse(date_str, languages=[language])
    if not parsed_date:
        return "[DATE]" if language == "en" else "[DATUM]"

    year = parsed_date.year

    # 用更宽泛的分隔符拆分字符串以检查是否包含月份数字
    parts = re.split(r'[\s\-/\.]+', date_str)
    has_month = any(part.isdigit() and 1 <= int(part) <= 12 for part in parts)

    if level == "mild":
        result = f"{year}-{parsed_date.month:02d}" if has_month else f"{year}-XX"
    elif level == "moderate":
        result = f"{year}"
    elif level == "severe":
        min_year = year - 2
        max_year = year + 8
        result = f"{min_year}-{max_year}"
    else:
        return "[DATE]" if language == "en" else "[DATUM]"

    # 根据语言附加特定的后缀
    if language == "de":
        if level == "mild":
            suffix = " (Jahr-Monat)" if has_month else " (Jahr-XX)"
        elif level == "moderate":
            suffix = " (Jahr)"
        elif level == "severe":
            suffix = " (ungefähr)"
        else:
            suffix = ""
    else:
        if level == "mild":
            suffix = " (Year-Month)" if has_month else " (Year-XX)"
        elif level == "moderate":
            suffix = " (Year)"
        elif level == "severe":
            suffix = " (approx.)"
        else:
            suffix = ""
    return result + suffix


def do_replacement(orig, start, end, replacement):
    if start < 0 or end < 0 or start >= end:
        return orig
    left_part = orig[:start]
    right_part = orig[end:]
    return left_part + " " + replacement + right_part


def process_text_by_paragraphs(original_text, nlp_pipeline):
    # 按换行符分割段落
    paragraphs = original_text.split('\n')

    # 去除空段落（如果有）
    paragraphs = [para.strip() for para in paragraphs if para.strip()]

    # 对每个段落进行NER处理
    ner_results = []
    offset = 0  # 用来记录当前段落在整个文本中的起始偏移量

    for paragraph in paragraphs:
        result = nlp_pipeline(paragraph)

        # 对每个实体，调整其起始位置和结束位置
        for entity in result:
            entity['start'] += offset  # 调整实体的起始位置
            entity['end'] += offset    # 调整实体的结束位置

        ner_results.append(result)

        # 更新偏移量（每个段落的长度加上换行符）
        offset += len(paragraph) + 2  # 加1是为了包含换行符

    # 合并结果
    merged_results = []
    for result in ner_results:
        merged_results.extend(result)  # 将每个段落的结果合并成一个列表

    return merged_results


@app.route('/')
@app.route('/ner')
def ner():
    return render_template('NER.html', title="NER - Medical Data Anonymization")


@app.route('/deidentification')
def deidentification():
    return render_template('De-identification.html', title="De-identification - Medical Data Anonymization")


@app.route('/deidentification/file')
def deidentification_file():
    """
    Render the De-identification File page.
    """
    return render_template('De-identification-file.html', title="De-identification - File Upload")


@app.route('/about')
def about():
    return render_template('About.html', title="About - Medical Data Anonymization")


@app.route('/process', methods=['POST'])
def process():
    try:
        text = request.form.get('text', '')
        threshold = float(request.form.get('threshold', 0))
        ner_results = process_text_by_paragraphs(text, nlp_pipeline)

        if not ner_results:
            return jsonify({
                'result': text,
                'overall_confidence': 0.0,
                'privacy_score': 0.0
            })

        segments = []
        current_pos = 0
        confidence_scores = []

        ner_results = sorted(ner_results, key=lambda x: x['start'])

        for entity in ner_results:
            start, end = entity['start'], entity['end']
            label = entity['entity_group']
            confidence = float(entity.get('score', 0))
            label_color = assign_color(label)
            confidence_color = get_confidence_color(confidence)

            if current_pos < start:
                segments.append({'text': text[current_pos:start], 'label': None})

            if confidence >= threshold:
                segments.append({
                    'text': text[start:end],
                    'label': label,
                    'color': label_color,
                    'score': confidence,
                    'confidence_color': confidence_color
                })
                confidence_scores.append(confidence)
            else:
                segments.append({'text': text[start:end], 'label': None})

            current_pos = end

        if current_pos < len(text):
            segments.append({'text': text[current_pos:], 'label': None})

        overall_confidence = (
            float(sum(confidence_scores)) / len(confidence_scores) if confidence_scores else 0.0
        )

        privacy_score = calculate_privacy_score(ner_results)

        result_str = ""
        for segment in segments:
            if segment['label']:
                result_str += (
                    f'<span style="background-color:{segment["color"]}; '
                    f'padding: 5px 10px; margin: 2px; border-radius: 25px; '
                    f'box-shadow: 0px 1px 5px rgba(0, 0, 0, 0.2); font-weight: bold;" '
                    f'title="Confidence: {segment["score"]:.2f}" '
                    f'data-confidence-color="{segment["confidence_color"]}">'
                )
                result_str += (
                    f'{segment["text"]} '
                    f'<span style="background-color:rgba(0, 0, 0, 0.1); '
                    f'color: #fff; padding: 2px 4px; border-radius: 5px; font-size: 0.8em;">'
                    f'{segment["label"]}</span>'
                )
                result_str += '</span>'
            else:
                result_str += segment['text']

        if not result_str:
            result_str = text

        return jsonify({
            'result': result_str,
            'overall_confidence': overall_confidence,
            'privacy_score': privacy_score
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'result': text,
            'overall_confidence': 0.0,
            'privacy_score': 0.0
        }), 500


@app.route('/process_deid', methods=['POST'])
def process_deid():
    text = request.form['text']
    ner_results = process_text_by_paragraphs(text, nlp_pipeline)
    privacy_score_before = calculate_privacy_score(ner_results)

    entities = []
    for entity in ner_results:
        entities.append({
            'entity_group': entity['entity_group'],
            'start': entity['start'],
            'end': entity['end'],
            'text': text[entity['start']:entity['end']]
        })

    entity_groups = list(set([e['entity_group'] for e in entities]))

    return jsonify({
        'entities': entities,
        'entity_groups': entity_groups,
        'original_text': text,
        'privacy_score_before': privacy_score_before
    })


@app.route('/apply_deid', methods=['POST'])
def apply_deid():
    data = request.get_json()
    original_text = data['original_text']
    strategies = data['strategies']
    entities = data['entities']
    threshold = float(data.get('threshold', 0))
    language = data['language']

    replaced_text = original_text

    pseudonym_counters = {}

    for ent in sorted(entities, key=lambda x: x['start'], reverse=True):
        eg = ent['entity_group']
        confidence = float(ent.get('confidence', 0))

        if eg not in strategies or confidence < threshold:
            continue

        strategy = strategies[eg]
        start, end = ent['start'], ent['end']
        text_slice = ent['text']

        if strategy == "delete":
            replaced_text = do_replacement(replaced_text, start, end, "[ZENSIERT]" if language == "de" else "[REDACTED]")
        elif strategy == "pseudonymize":
            if eg not in pseudonym_counters:
                pseudonym_counters[eg] = 0
            pseudonym_counters[eg] += 1
            letter = chr(65 + (pseudonym_counters[eg] - 1) % 26)
            replaced_text = do_replacement(replaced_text, start, end, f"{eg}_{letter}")
        elif strategy.startswith("generalize"):
            level = strategy.split("_")[1]
            if eg == "AGE":
                generalized = generalize_age(text_slice, level, language)
            elif eg == "DATE":
                generalized = generalize_date(text_slice, level, language)
            else:
                generalized = "[GENERALIERT]" if language == "de" else "[GENERALIZED]"
            replaced_text = do_replacement(replaced_text, start, end, generalized)

    replaced_text = " ".join(replaced_text.split())
    ner_results_after = process_text_by_paragraphs(replaced_text, nlp_pipeline)
    privacy_score_after = calculate_privacy_score(ner_results_after)

    return jsonify({
        'deidentified_text': replaced_text,
        'privacy_score_after': privacy_score_after
    })


@app.route('/process_file', methods=['POST'])
def process_file():
    try:
        # 获取上传的文件
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # 获取传递的列名、阈值和实体策略
        selected_column = request.form.get('columns', '')
        threshold = float(request.form.get('threshold', 0))
        entity_strategies = json.loads(request.form.get('entity_strategies', '{}'))
        selected_column = selected_column.strip()
        selected_column = selected_column.replace('\r\n', '')
        language = request.form.get('language', 'en')

        # 打印接收到的字段，用于调试
        print(f"Selected column: {selected_column}")
        print(f"Threshold: {threshold}")
        print(f"Entity strategies: {entity_strategies}")

        # 读取CSV文件
        df = pd.read_csv(io.BytesIO(file.read()))

        # 检查选中的列是否存在
        if selected_column not in df.columns:
            return jsonify({'error': f"Column '{selected_column}' not found in the CSV file"}), 400

        # 根据用户选择的列进行去标识化处理
        results = []

        total_privacy_risk_level_before = 0
        total_privacy_risk_level_after = 0
        count = 0

        for index, row in df.iterrows():
            # 获取原始文本（假设选中的列）
            original_text = row.get(selected_column, '')  # 使用正确的列名
            if not original_text:
                continue  # 如果没有文本，则跳过

            # 获取命名实体识别（NER）结果
            ner_results = process_text_by_paragraphs(original_text, nlp_pipeline)

            # 如果没有实体，跳过此行
            if not ner_results:
                continue
            privacy_risk_level_before = calculate_privacy_score(ner_results)

            # 进行实体去标识化
            replaced_text = original_text
            pseudonym_counters = {}

            # 处理每个实体，根据其策略进行去标识化
            # 需要按 start 从后往前排序，避免覆盖
            for entity in sorted(ner_results, key=lambda x: x['start'], reverse=True):  # 重要：按start从后向前处理
                entity_group = entity['entity_group']
                confidence = float(entity.get('score', 0))

                # 检查实体组是否在策略中
                if entity_group not in entity_strategies or confidence < threshold:
                    continue

                # 获取该实体的去标识化策略
                strategy = entity_strategies[entity_group]
                start, end = entity['start'], entity['end']
                text_slice = original_text[start:end]

                # 根据策略进行处理
                if strategy == "delete":
                    if language == "en":
                        replaced_text = do_replacement(replaced_text, start, end, "[REDACTED]")
                    else:
                        replaced_text = do_replacement(replaced_text, start, end, "[ZENSIERT]")
                elif strategy == "pseudonymize":
                    if entity_group not in pseudonym_counters:
                        pseudonym_counters[entity_group] = 0
                    pseudonym_counters[entity_group] += 1
                    letter = chr(65 + (pseudonym_counters[entity_group] - 1) % 26)
                    # 根据语言可以调整伪名格式，这里暂时保持不变
                    replacement = f"{entity_group}_{letter}"
                    replaced_text = do_replacement(replaced_text, start, end, replacement)
                elif strategy.startswith("generalize"):
                    level = strategy.split("_")[1]
                    if entity_group == "AGE":
                        generalized = generalize_age(text_slice, level, language)
                    elif entity_group == "DATE":
                        generalized = generalize_date(text_slice, level, language)
                    else:
                        generalized = "[GENERALIZED]" if language == "en" else "[GENERALIERT]"
                    replaced_text = do_replacement(replaced_text, start, end, generalized)

            results.append({
                'De-identified_text': replaced_text
            })

            # Calculate risk level
            ner_results_after = process_text_by_paragraphs(replaced_text, nlp_pipeline)
            privacy_risk_level_after = calculate_privacy_score(ner_results_after)
            total_privacy_risk_level_before += privacy_risk_level_before
            total_privacy_risk_level_after += privacy_risk_level_after
            count += 1

        # 确保结果不为空
        if not results:
            print("No valid data was processed. Returning empty result.")
            return jsonify({'error': 'No valid data was processed.'}), 400
        # 计算平均值
        if count > 0:
            avg_privacy_risk_level_before = total_privacy_risk_level_before / count
            avg_privacy_risk_level_after = total_privacy_risk_level_after / count
        else:
            avg_privacy_risk_level_before = 0
            avg_privacy_risk_level_after = 0

        # 输出平均值
        print(f"Average Privacy Risk Level Before De-identification: {avg_privacy_risk_level_before}")
        print(f"Average Privacy Risk Level After De-identification: {avg_privacy_risk_level_after}")

        # 将去标识化结果转换为CSV格式并返回
        result_df = pd.DataFrame(results)
        output = io.StringIO()
        result_df.to_csv(output, index=False)
        output.seek(0)

        # 返回去标识化后的CSV内容
        return jsonify({
            'result': output.getvalue(),
            'avg_privacy_risk_level_before': avg_privacy_risk_level_before,
            'avg_privacy_risk_level_after': avg_privacy_risk_level_after
        })

    except Exception as e:
        print(f"Error: {e}")  # 打印详细错误信息
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

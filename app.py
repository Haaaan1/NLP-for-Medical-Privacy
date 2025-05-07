from flask import Flask, render_template, request, jsonify, send_file
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


def calculate_privacy_risk_level(entities):
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
    # Use the provided language parameter without auto-detecting language
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

        # Append language-specific suffix
        if language == "de":
            return f"{lower_bound}-{upper_bound} Jahre alt"
        else:
            return f"{lower_bound}-{upper_bound} years old"

    generalized_str = pattern.sub(replace_age, age_str)
    if generalized_str == age_str:
        return "[ALTER]" if language == "de" else "[AGE]"
    return generalized_str


def generalize_date(date_str, level="mild", language="en"):
    # Parse the date using dateparser with the specified language
    parsed_date = dateparser.parse(date_str, languages=[language])
    if not parsed_date:
        return "[DATE]" if language == "en" else "[DATUM]"

    year = parsed_date.year
    # Split the string using a broad set of delimiters to check for month presence
    parts = re.split(r'[\s\-/\.]+', date_str)
    has_month = any(part.isdigit() and 1 <= int(part) <= 12 for part in parts)

    if level == "mild":
        result = f"{year}-{parsed_date.month:02d}" if has_month else f"{year}"
    elif level == "moderate":
        result = f"{year}"
    elif level == "severe":
        min_year = year - 2
        max_year = year + 8
        result = f"{min_year}-{max_year}"
    else:
        return "[DATE]" if language == "en" else "[DATUM]"

    return result


def do_replacement(orig, start, end, replacement):
    if start < 0 or end < 0 or start >= end:
        return orig
    left_part = orig[:start]
    right_part = orig[end:]
    return left_part + " " + replacement + right_part


def process_text_by_paragraphs(original_text, nlp_pipeline):
    # Split text by newline and remove empty paragraphs
    paragraphs = [para.strip() for para in original_text.split('\n') if para.strip()]

    ner_results = []
    offset = 0  # Track the starting offset for each paragraph in the entire text

    for paragraph in paragraphs:
        result = nlp_pipeline(paragraph)
        # Adjust each entity's start and end positions to the overall text
        for entity in result:
            entity['start'] += offset
            entity['end'] += offset
        ner_results.append(result)
        # Update offset (accounting for paragraph length and newline characters)
        offset += len(paragraph) + 2

    # Merge results from all paragraphs into a single list
    merged_results = []
    for result in ner_results:
        merged_results.extend(result)

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

        privacy_score = calculate_privacy_risk_level(ner_results)

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
    privacy_score_before = calculate_privacy_risk_level(ner_results)

    entities = []
    for entity in ner_results:
        confidence = float(entity.get('score', 0))
        entities.append({
            'entity_group': entity['entity_group'],
            'start': entity['start'],
            'end': entity['end'],
            'text': text[entity['start']:entity['end']],
            'score': confidence
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
        confidence = float(ent.get('score', 0))
        print(eg)
        print(confidence)
        print(threshold)
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
    privacy_score_after = calculate_privacy_risk_level(ner_results_after)

    return jsonify({
        'deidentified_text': replaced_text,
        'privacy_score_after': privacy_score_after
    })


@app.route('/process_file', methods=['POST'])
def process_file():
    try:
        # Get the uploaded file
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Get selected column, threshold, and entity strategies
        selected_column = request.form.get('columns', '')
        threshold = float(request.form.get('threshold', 0))
        entity_strategies = json.loads(request.form.get('entity_strategies', '{}'))
        selected_column = selected_column.strip()
        selected_column = selected_column.replace('\r\n', '')
        language = request.form.get('language', 'en')

        # Read CSV file
        df = pd.read_csv(io.BytesIO(file.read()))

        # Check if the selected column exists
        if selected_column not in df.columns:
            return jsonify({'error': f"Column '{selected_column}' not found in the CSV file"}), 400

        # Create result DataFrame with all original columns preserved
        result_df = df.copy()

        total_privacy_risk_level_before = 0
        total_privacy_risk_level_after = 0
        count = 0

        for index, row in df.iterrows():
            original_text = row.get(selected_column, '')
            if pd.isna(original_text) or not str(original_text).strip():
                continue

            # Get NER results
            ner_results = process_text_by_paragraphs(str(original_text), nlp_pipeline)

            if not ner_results:
                continue

            privacy_risk_level_before = calculate_privacy_risk_level(ner_results)
            replaced_text = str(original_text)
            pseudonym_counters = {}

            # Process each entity
            for entity in sorted(ner_results, key=lambda x: x['start'], reverse=True):
                entity_group = entity['entity_group']
                confidence = float(entity.get('score', 0))

                if entity_group not in entity_strategies or confidence < threshold:
                    continue

                strategy = entity_strategies[entity_group]
                start, end = entity['start'], entity['end']
                text_slice = str(original_text)[start:end]

                if strategy == "delete":
                    replacement = "[REDACTED]" if language == "en" else "[ZENSIERT]"
                    replaced_text = do_replacement(replaced_text, start, end, replacement)
                elif strategy == "pseudonymize":
                    if entity_group not in pseudonym_counters:
                        pseudonym_counters[entity_group] = 0
                    pseudonym_counters[entity_group] += 1
                    letter = chr(65 + (pseudonym_counters[entity_group] - 1) % 26)
                    replacement = f"{entity_group}_{letter}"
                    replaced_text = do_replacement(replaced_text, start, end, replacement)
                elif strategy.startswith("generalize"):
                    level = strategy.split("_")[1]
                    if entity_group == "AGE":
                        replacement = generalize_age(text_slice, level, language)
                    elif entity_group == "DATE":
                        replacement = generalize_date(text_slice, level, language)
                    else:
                        replacement = "[GENERALIZED]" if language == "en" else "[GENERALIERT]"
                    replaced_text = do_replacement(replaced_text, start, end, replacement)

            # Update result DataFrame
            result_df.at[index, selected_column] = replaced_text

            # Recalculate risk level
            ner_results_after = process_text_by_paragraphs(replaced_text, nlp_pipeline)
            privacy_risk_level_after = calculate_privacy_risk_level(ner_results_after)
            total_privacy_risk_level_before += privacy_risk_level_before
            total_privacy_risk_level_after += privacy_risk_level_after
            count += 1

        if count == 0:
            return jsonify({'error': 'No valid data was processed.'}), 400

        # Calculate average risk levels
        avg_privacy_risk_level_before = total_privacy_risk_level_before / count
        avg_privacy_risk_level_after = total_privacy_risk_level_after / count

        # Convert result to CSV string
        output = io.StringIO()
        result_df.to_csv(output, index=False)
        csv_content = output.getvalue()

        # Return response in frontend-compatible format
        return jsonify({
            'result': csv_content,  # Keep as CSV string
            'avg_privacy_risk_level_before': avg_privacy_risk_level_before,
            'avg_privacy_risk_level_after': avg_privacy_risk_level_after,
            '_original_filename': file.filename
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

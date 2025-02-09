from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import random
import re
from datetime import datetime
from dateutil import parser

app = Flask(__name__)

# Load model and pipeline
nlp_pipeline = pipeline(
    "token-classification",
    model="Clinical-AI-Apollo/Medical-NER",
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


@app.route('/')
@app.route('/ner')
def ner():
    """
    Render the NER page.
    """
    return render_template('NER.html', title="NER - Medical Data Anonymization")


@app.route('/deidentification')
def deidentification():
    """
    Render the De-identification page.
    """
    return render_template('De-identification.html', title="De-identification - Medical Data Anonymization")


@app.route('/about')
def about():
    """
    Render the About page.
    """
    return render_template('About.html', title="About - Medical Data Anonymization")


# Assign color for new labels if they are not in LABEL_COLORS
def assign_color(label):
    if label not in LABEL_COLORS:
        if len(LABEL_COLORS) < len(COLOR_PALETTE):
            LABEL_COLORS[label] = COLOR_PALETTE[len(LABEL_COLORS) % len(COLOR_PALETTE)]
        else:
            LABEL_COLORS[label] = f'#{random.randint(0, 0xFFFFFF):06x}'
    return LABEL_COLORS[label]


@app.route('/process', methods=['POST'])
def process():
    """
    Handle NER highlighting for the input text,
    filter entities based on confidence threshold, and return the result.
    Also calculate the overall confidence score for displayed entities.
    """
    text = request.form['text']
    threshold = float(request.form.get('threshold', 0))  # Default threshold is 0
    ner_results = nlp_pipeline(text)  # Assume nlp_pipeline returns results

    segments = []
    current_pos = 0
    confidence_scores = []

    def assign_color(label):
        """
        Assign color for each label.
        """
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

    ner_results = sorted(ner_results, key=lambda x: x['start'])

    for entity in ner_results:
        start, end = entity['start'], entity['end']
        label = entity['entity_group']
        confidence = float(entity['score'])  # Ensure confidence is a Python float
        label_color = assign_color(label)
        confidence_color = get_confidence_color(confidence)

        if current_pos < start:
            segments.append({'text': text[current_pos:start], 'label': None})

        # Only annotate entities with confidence >= threshold
        if confidence >= threshold:
            segments.append({
                'text': text[start:end],
                'label': label,
                'color': label_color,
                'score': confidence,
                'confidence_color': confidence_color
            })
            confidence_scores.append(confidence)  # Include in confidence score calculation
        else:
            segments.append({'text': text[start:end], 'label': None})

        current_pos = end

    if current_pos < len(text):
        segments.append({'text': text[current_pos:], 'label': None})

    # Calculate overall confidence score
    overall_confidence = (
        float(sum(confidence_scores)) / len(confidence_scores) if confidence_scores else 0.0
    )

    # Convert segments into HTML
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

    return jsonify({'result': result_str, 'overall_confidence': overall_confidence})


@app.route('/process_deid', methods=['POST'])
def process_deid():
    """
    For de-identification page:
    1) Run NER on the input text.
    2) Build a list of recognized entities.
    3) Return them to the front-end for user strategy selection.
    """
    text = request.form['text']
    ner_results = nlp_pipeline(text)

    # Collect entities in a structured format
    entities = []
    for entity in ner_results:
        entities.append({
            'entity_group': entity['entity_group'],
            'start': entity['start'],
            'end': entity['end'],
            'text': text[entity['start']:entity['end']]
        })

    # Also return all distinct entity groups
    entity_groups = list(set([e['entity_group'] for e in entities]))

    return jsonify({
        'entities': entities,
        'entity_groups': entity_groups,
        'original_text': text
    })


##############################
# Helper functions for age/date generalization
##############################


def generalize_age(age_str, level="mild"):
    """
    Generalize age numbers within a string using the specified level.
    Only age numbers (e.g., "17-year-old", "17 years old", "age 17") are modified.
    The rest of the string is preserved.

    Parameters:
    - age_str: str, input string containing age information.
    - level: str, one of "mild", "moderate", "severe".

    Returns:
    - str, the age generalized string. Returns "[AGE]" if no valid age is found.
    """
    # Define the regular expression pattern:
    # 1. Match numbers followed by age-related words.
    # 2. Match "age" prefix followed by numbers, with possible ":" or "-" separators.
    pattern = re.compile(
        r'\b(\d+)\s*(?:-year-old|years old|years-old|yr-old|years|year|yrs)\b'
        r'|\bage\s*[:\-]?\s*(\d+)\b',
        re.IGNORECASE
    )

    def replace_age(match):
        # match.group(1): Case where number is followed by age-related words.
        # match.group(2): Case where "age" is followed by a number.
        num_str = match.group(1) if match.group(1) else match.group(2)
        if not num_str:
            return "[AGE]"
        try:
            original_age = int(num_str)
        except ValueError:
            return "[AGE]"

        # Determine the interval based on the level
        if level == "mild":
            interval = 5
        elif level == "moderate":
            interval = 10
        elif level == "severe":
            interval = 20
        else:
            interval = 5  # Default to mild

        # Calculate the lower and upper bounds
        lower_bound = (original_age // interval) * interval
        upper_bound = lower_bound + interval

        # Ensure bounds are not negative
        lower_bound = max(lower_bound, 0)
        upper_bound = max(upper_bound, 0)

        # Return the age range as a string
        return f"{lower_bound}-{upper_bound}"

    # Use re.sub to perform the replacement
    generalized_str = pattern.sub(replace_age, age_str)

    # If no replacement occurred, return the default value "[AGE]"
    if generalized_str == age_str:
        return "[AGE]"
    return generalized_str


def generalize_date(date_str, level="mild"):
    """
    Generalize a date string with various levels of obfuscation:
      - mild: Keep 'year-month' if available (e.g., "2022-04"), or "2022-XX" if no month is provided.
      - moderate: Keep only the year (e.g., "2022").
      - severe: Transform to a wide year range (e.g., "2020-2030").
    
    Handles multiple input formats and partial dates.
    """
    try:
        # Try parsing the date using dateutil
        parsed_date = parser.parse(date_str, fuzzy=True)
        year = parsed_date.year

        # Check if the original string contains month information
        has_month = any(part.isdigit() and 1 <= int(part) <= 12 for part in date_str.split('-'))

        if level == "mild":
            # Generalize to "year-month" if month exists; otherwise, "year-XX"
            return f"{year}-{parsed_date.month:02d}" if has_month else f"{year}-XX"
        elif level == "moderate":
            # Generalize to year only
            return f"{year}"
        elif level == "severe":
            # Create a wide range for year
            min_year = year - 2
            max_year = year + 8
            return f"{min_year}-{max_year}"
        else:
            return "[DATE]"  # Default fallback
    except ValueError:
        # Fallback if the date string cannot be parsed
        return "[DATE]"


@app.route('/apply_deid', methods=['POST'])
def apply_deid():
    data = request.get_json()
    original_text = data['original_text']
    strategies = data['strategies']
    entities = data['entities']
    threshold = float(data.get('threshold', 0))  # 默认阈值为0

    replaced_text = original_text

    # Counters for pseudonymize
    pseudonym_counters = {}

    def do_replacement(orig, start, end, replacement):
        if start < 0 or end < 0 or start >= end:
            return orig
        left_part = orig[:start]
        right_part = orig[end:]
        # 在替换内容前添加一个空格
        return left_part + " " + replacement + right_part

    # Sort entities in reverse order
    for ent in sorted(entities, key=lambda x: x['start'], reverse=True):
        eg = ent['entity_group']
        confidence = float(ent.get('confidence', 0))  # 从实体中获取置信度

        if eg not in strategies or confidence < threshold:
            # 跳过不符合置信度阈值的实体
            continue

        strategy = strategies[eg]
        start, end = ent['start'], ent['end']
        text_slice = ent['text']

        if strategy == "delete":
            replaced_text = do_replacement(replaced_text, start, end, "[REDACTED]")
        elif strategy == "pseudonymize":
            if eg not in pseudonym_counters:
                pseudonym_counters[eg] = 0
            pseudonym_counters[eg] += 1
            letter = chr(65 + (pseudonym_counters[eg] - 1) % 26)  # 循环生成 'A', 'B', ...
            replaced_text = do_replacement(replaced_text, start, end, f"{eg}_{letter}")
        elif strategy.startswith("generalize"):
            level = strategy.split("_")[1]
            if eg == "AGE":
                generalized = generalize_age(text_slice, level)
            elif eg == "DATE":
                generalized = generalize_date(text_slice, level)
            else:
                generalized = "[GENERALIZED]"
            replaced_text = do_replacement(replaced_text, start, end, generalized)

    # 清理多余的空格（例如连续的空格）
    replaced_text = " ".join(replaced_text.split())

    return jsonify({'deidentified_text': replaced_text})


if __name__ == '__main__':
    app.run(debug=True)

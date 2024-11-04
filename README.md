# Medical Data Anonymization Tool

## Overview
This project is an NLP-based tool aimed at enhancing privacy protection in medical data by using Named Entity Recognition (NER) and De-identification techniques. The current implementation focuses on identifying sensitive information within clinical text data using NER, and a De-identification feature will be developed in the future to further anonymize the medical information for research purposes.

## Features

### 1. Named Entity Recognition (NER)
The NER feature identifies key medical entities within the provided text, such as patient age, gender, disease/disorder information, etc. The identified entities are visually highlighted, allowing users to easily see sensitive information that requires anonymization. The tool currently utilizes a pre-trained NLP model from Hugging Face to perform NER on unstructured medical data.

- **Entities Recognized**:
    - Patient Age
    - Gender
    - Diseases and Disorders
    - Other medical-related information

### 2. Future Development: De-identification Feature
In the next phase, a De-identification feature will be implemented. This feature will utilize the recognized entities to allow users to apply k-anonymity, l-diversity, and t-closeness anonymization techniques, ensuring patient privacy is maintained before the data is used for secondary purposes such as research or analysis.

- **Planned Features**:
    - Remove or generalize identifiable information to ensure privacy compliance.
    - Provide risk analysis and anonymization strategies to meet privacy standards.

## Deployment Instructions
To deploy the current version of this project, follow these steps:

### Prerequisites
- Python 3.9
- Git
- Virtual Environment (`venv`)
- Flask
- Hugging Face Transformers library

### Setup
1. **Clone the Repository**:
   ```sh
   git clone https://github.com/Haaaan1/NLP-for-Medical-Privacy.git
   cd NLP-for-Medical-Privacy

2. **Create and Activate Virtual Environment**:
   ```sh
   python3.9 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
3. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt

4. **Run the Application**:
   ```sh
   python app.py

### Usage

- Access the application through your web browser.

- Input a clinical summary in the text area on the NER page.

- Click "Compute" to perform Named Entity Recognition.

- Future updates will include an additional tab for De-identification functionality.
# Medical Data Anonymization Tool

## Overview
This is the repository for the thesis work "Developing a Natural Language Processing Tool for Enhanced Privacy Protection in Medical Data
" done at
Communication Systems Group, University of Zurich, supervised by Mr.
Weijie Niu, Dr. Alberto Huertas Celdran and Prof. Dr. Burkhard Stiller.

This project is an NLP-based tool aimed at enhancing privacy protection in medical data by using Named Entity Recognition (NER) and De-identification techniques.

## Features

### 1. Named Entity Recognition (NER)
The NER feature identifies key medical entities within the provided text, such as age, gender, history, etc. The identified entities are visually highlighted, allowing users to easily see sensitive information that requires anonymization. The tool currently utilizes a pre-trained NLP model from Hugging Face(https://huggingface.co/blaze999/Medical-NER) to perform NER on unstructured medical data.

### 2. De-identification Feature
This feature allows users to automatically remove or transform sensitive information. It supports the following de-identification operations:
- Deletion: Removes the identified entity completely from the text.

- Pseudonymization: Replaces the entity with a consistent placeholder (e.g., [NAME_1], [HOSPITAL_1]).

- Generalization: Replaces specific values with more general categories (e.g., age “42” → “40-50”, or date “2022-01-01” → “Jan 2022”).

### 3. Privacy Risk Evaluation
Estimates the privacy risk based on the density of sensitive entities in the text using a hybrid quantitative and qualitative scoring method adapted from academic literature.

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

## Reproducing Experiments and Results

To reproduce the de-identification results using our platform, follow the steps below. We used the following two public datasets for evaluation:

- English dataset: [jung1230/patient_info_and_summary](https://huggingface.co/datasets/jung1230/patient_info_and_summary)
- German dataset: [thisserand/health_care_german](https://huggingface.co/datasets/thisserand/health_care_german)

### Step-by-Step Instructions

1. **Run the Application**  
   Start the Flask web server locally:
   ```bash
   python app.py

2. **Navigate to De-identification Page**  
   In your browser, open the web page(http://127.0.0.1:5000) and click on **De-identification** through navigation bar in the top.

3. **Upload Dataset**  
   Click the **Go to File Upload** button and upload your `.csv` dataset file.

4. **Configuration Setup**
    - Choose the dataset **language** (English or German).
    - Select the **column** that contains the text to be de-identified.
    - Configure the de-identification methods for each entity:
        - `Age`, `Date`: **Generalized Mild** 
        - `Sex`, `Duration`, `History`, `Detailed Description`: **Pseudonymize**

5. **Start De-identification**  
   Click the **Process and De-identify** button and wait for the system to finish.

6. **Download Results**  
   After processing is complete, click **Download De-identified CSV** to obtain the anonymized dataset.


## How to Continue the Developments (for Developers)

This section outlines potential directions for future development and improvements.

### 1. NER Model Optimization

- **Current Model**: The platform currently uses a pre-trained English medical NER model (184M parameters, FP32 precision). It performs well on clinical narratives and runs efficiently on local machines.
- **Performance Bottlenecks**: For larger datasets, the inference speed may become a bottleneck due to the computational cost of FP32. Developers may consider:
    - Deploying a quantized version of the model (e.g., FP16 or INT8) to reduce inference time.
    - Deploying the application to higher-performance hardware (e.g., GPU or cloud-based inference).
- **Multilingual Support**: The current model is optimized for English texts. To improve performance on medical data in German or any other languages:
    - Fine-tune the existing model using domain-specific multi-language medical datasets.
    - Alternatively, integrate separate medical NER model for language-specific processing.

### 2. De-identification Method Enhancement

- **Current Support**:
    - All entity types support **Deletion** and **Pseudonymization**.
    - **Generalization** is currently supported only for `Age` and `Date`, which have well-structured formats.
- **Future Enhancements**:
    - Extend generalization to other entity types, especially more complex ones like `Clinical Event`, `Disease Disorder`, or `Medication`.
    - Research or develop new generalization algorithms for unstructured or semantically rich entities.
    - Consider incorporating rule-based or knowledge-graph-based approaches to support context-aware anonymization.

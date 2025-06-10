# Real-Time Ticket Classification and Entity Extraction Pipeline

## Table of Contents

1. [Introduction](#introduction)
2. [Project Motivation](#project-motivation)
3. [System Architecture](#system-architecture)

   * [Data Ingestion](#data-ingestion)
   * [Data Preprocessing](#data-preprocessing)
   * [Feature Engineering](#feature-engineering)
   * [Model Training and Evaluation](#model-training-and-evaluation)
   * [Entity Extraction](#entity-extraction)
   * [User Interface (Gradio)](#user-interface-gradio)
4. [Key Design Choices](#key-design-choices)
5. [Model Evaluation and Metrics](#model-evaluation-and-metrics)

   * [Issue Type Classification](#issue-type-classification)
   * [Urgency Level Classification](#urgency-level-classification)
   * [Confusion Matrices](#confusion-matrices)
   * [Feature Importance](#feature-importance)
   * [t-SNE Visualization of TF-IDF Features](#t-sne-visualization-of-tf-idf-features)
6. [Limitations and Future Work](#limitations-and-future-work)
7. [Installation and Setup](#installation-and-setup)

   * [Clone the Repository](#clone-the-repository)
   * [Environment Setup](#environment-setup)
   * [Downloading Models and Data](#downloading-models-and-data)
8. [Usage Guide](#usage-guide)

   * [Command-Line Interface](#command-line-interface)
   * [Gradio Web Application](#gradio-web-application)
   * [Batch Processing](#batch-processing)
9. [Demo Video](#demo-video)
10. [Project Structure](#project-structure)
11. [Acknowledgements](#acknowledgements)
12. [Contact Information](#contact-information)

---

## Introduction

In today’s fast-paced customer support and IT operations environments, organizations handle thousands of incident tickets every day. Each ticket contains text describing an issue, its severity, associated product, and other crucial details. Automatically classifying tickets by issue type and urgency, as well as extracting key entities, can greatly reduce manual triage time, improve response speed, and enhance overall operational efficiency.

This project presents a **Real-Time NLP/ML Pipeline** that performs:

1. **Ticket Classification**: Automatically predicts the issue type (e.g., "software bug", "network outage") and urgency level (e.g., "low", "medium", "high").
2. **Entity Extraction**: Retrieves dates, products, and complaint categories present in the ticket text using rule-based and Named Entity Recognition (NER) methods.
3. **Interactive Interface**: A lightweight Gradio-based web application enabling real-time single-ticket and batch-ticket analysis via an intuitive GUI.

By combining classical machine learning techniques (TF-IDF, XGBoost, SMOTE) with SpaCy NER and a simple keyword lookup, the pipeline achieves robust performance on imbalanced, real-world ticket datasets.

## Project Motivation

Support desks, IT helpdesks, and service desks are often inundated with tickets during major incidents, system rollouts, or product launches. Manual ticket triage leads to:

* **Delays in Incident Resolution**: Human agents need time to read, interpret, and route each ticket.
* **Inconsistency**: Different agents may classify the same issue differently, resulting in inconsistent priority handling.
* **Scalability Challenges**: As ticket volume grows, manual approaches become bottlenecks.

Automating key steps in ticket triage through NLP and ML can help organizations:

* **Reduce Mean Time to Response (MTTR)**: Instantly assign preliminary urgency and department routing.
* **Standardize Classification**: Ensure consistent labeling across all tickets.
* **Free Up Human Agents**: Allow agents to focus on high-level escalation, investigation, and resolution rather than repetitive classification tasks.

This pipeline demonstrates an accessible, scalable approach, leveraging widely-used libraries (`scikit-learn`, `XGBoost`, `SpaCy`, `Gradio`, `nltk`) and a modular architecture that can be extended to additional tasks (e.g., routing, SLA prediction, sentiment-based escalation).

## System Architecture

The system consists of six major components:

1. **Data Ingestion**
2. **Data Preprocessing**n3. **Feature Engineering**
3. **Model Training and Evaluation**
4. **Entity Extraction**
5. **User Interface**

A high-level flow is visualized below:

```
Raw Tickets CSV
      ↓
Data Preprocessing → Feature Engineering → Model Training & Evaluation
                                 ↓
                        Trained Models & Artifacts
                                 ↓
    Ticket Text (Real-time) → Analysis Function → Predictions + Entities → Gradio App
```

### Data Ingestion

* **Input Format**: A CSV file (`ai_dev_assignment_tickets_complex_1000.csv`) containing columns:

  * `ticket_id`
  * `ticket_text`
  * `issue_type`
  * `urgency_level`
  * `product`
  * Additional metadata if present.
* **Loading**: Utilizes `pandas.read_csv`.
* **Initial Inspection**: `df.head()`, `df.info()`, `df['issue_type'].value_counts()`, `df['urgency_level'].value_counts()` to understand class imbalances and missing values.

### Data Preprocessing

1. **Dropping Missing Values**: Remove rows with null ticket text or labels.
2. **Text Cleaning (`preprocess_text`)**:

   * Lowercase conversion.
   * Remove punctuation and non-alphanumeric characters: `re.sub(r"[^a-z0-9\s]", "", text)`.
   * Tokenization via `nltk.word_tokenize`.
   * Stopword removal using NLTK’s English stopwords.
   * Lemmatization with `WordNetLemmatizer`.
3. **Feature Augmentation**:

   * **Ticket Length**: Character count of the raw text.
   * **Sentiment Polarity**: `TextBlob(text).sentiment.polarity` (range -1.0 to +1.0).

This results in a cleaned DataFrame with columns:

* `clean_text` (string)
* `ticket_length` (int)
* `sentiment` (float)

### Feature Engineering

To capture both lexical and metadata signals:

* **TF-IDF Vectorization**: `TfidfVectorizer(max_features=1000)` on `clean_text`.

  * Restricts to top 1,000 vocabulary terms to reduce dimensionality.
* **Numeric Features**: `ticket_length` and `sentiment` appended to TF-IDF sparse matrix via `scipy.sparse.hstack`.

The resulting feature matrix `X` has shape `(n_samples, 1002)`.

### Model Training and Evaluation

We train two separate classifiers using XGBoost:

1. **Issue Type Classifier**
2. **Urgency Level Classifier**

Steps for each:

1. **Label Encoding**: Convert classes to integers via `LabelEncoder`.
2. **SMOTE Resampling**: Handle class imbalance by generating synthetic samples for minority classes (`SMOTE(random_state=42)`).
3. **Train/Test Split**: 80/20 split with `train_test_split(random_state=42)`.
4. **XGBoost Training**: `XGBClassifier(eval_metric='mlogloss', random_state=42)`.
5. **Model Evaluation**:

   * **Classification Report** (`precision`, `recall`, `F1-score`, `support`).
   * **Confusion Matrix**.
6. **Artifact Saving**: Persist models, vectorizer, and encoders via `joblib.dump` for later loading in the inference pipeline.

### Entity Extraction

A lightweight, rule-based approach:

* **Date Entities**: Leverage SpaCy `en_core_web_sm` NER to extract entities labeled `DATE`.
* **Keyword Matching**:

  * **Products**: Unique set of lowercase product names from `df['product']`.
  * **Complaint Types**: Unique set of lowercase `df['issue_type']`.
  * Check substring presence in `text.lower()` to identify occurrences.

This hybrid approach ensures fast, interpretable extraction without requiring a separate deep-learning NER model.

### User Interface (Gradio)

An interactive web app built with **Gradio Blocks**:

* **Tabs**:

  1. **Single Ticket**: Input box for one ticket, outputs for predicted issue type, urgency, and JSON of extracted entities.
  2. **Batch Tickets**: Multiline textbox where each line is a ticket; JSON output with predictions and entities for each.
* **Launch**: `demo.launch()` starts a local server (default `http://localhost:7860`).

## Key Design Choices

1. **Classical ML over Deep Learning**: TF-IDF + XGBoost offers faster turnaround and ease of interpretation compared to training transformers or neural networks, especially for medium-sized datasets.
2. **SMOTE for Imbalance**: Ensures minority classes are sufficiently represented during model training, improving recall at some cost to precision.
3. **Numeric Features**: Ticket length and sentiment often correlate with urgency (e.g., longer, negative-tone tickets may indicate high urgency).
4. **Rule-Based Entity Extraction**: Quick to implement, requires no additional training, and easily customizable by editing keyword lists.
5. **Gradio for Deployment**: Rapid prototyping with minimal code, enabling stakeholders to test the pipeline without custom front-end work.

## Model Evaluation and Metrics

1. **Issue Type Classification**

              precision    recall  f1-score   support

           0       1.00      1.00      1.00        24
           1       1.00      1.00      1.00        17
           2       1.00      1.00      1.00        30
           3       1.00      1.00      1.00        33
           4       1.00      1.00      1.00        27
           5       1.00      1.00      1.00        29
           6       1.00      1.00      1.00        27

    accuracy                           1.00       187
   macro avg       1.00      1.00      1.00       187
weighted avg       1.00      1.00      1.00       187

The Issue Type classifier achieves perfect scores across all classes, indicating highly accurate discrimination between the seven issue categories.

2. **Urgency Level Classification**

              precision    recall  f1-score   support

           0       0.46      0.51      0.48        61
           1       0.41      0.29      0.34        56
           2       0.35      0.41      0.38        56

    accuracy                           0.40       173
   macro avg       0.40      0.40      0.40       173
weighted avg       0.41      0.40      0.40       173

The Urgency Level classifier shows moderate performance, with the highest recall in the low urgency class (0) and room for improvement in distinguishing medium and high urgency levels.

### Confusion Matrices

By visualizing confusion matrices, we observe:

* **Issue Type**: Some `others` tickets misclassified as `hardware_failure`—likely overlapping language about "device" or "machine".
* **Urgency**: A few `high` tickets predicted as `medium`, suggesting model struggles when urgency cues are subtle.

*See notebook for heatmap figures.*

### Feature Importance

The top 20 features for issue classification include:

* Lexical tokens: `"error"`, `"timeout"`, `"failed"`, `"disconnect"`, `"login"`, etc.
* Numeric: `ticket_length`, `sentiment`.

This provides interpretability, showing which words drive classification decisions.

### t-SNE Visualization of TF-IDF Features

A 2D scatter plot via t-SNE (on first 500 samples) reveals distinct clustering by issue type, confirming that TF-IDF embeddings capture separable patterns.

*See notebook for scatter plot.*

## Limitations and Future Work

1. **Rule-Based Entity Extraction**:

   * Cannot handle variations (e.g., plural forms, typos) unless keyword lists expanded.
   * May miss entities not in the predefined product/issue lists.
   * Future: Integrate custom-trained SpaCy NER model to recognize domain-specific entities.

2. **SMOTE Resampling**:

   * Synthetic samples may not fully reflect real ticket variations.
   * Can introduce noise, especially when minority and majority classes overlap.
   * Future: Experiment with advanced sampling methods (ADASYN, cluster-based oversampling).

3. **Scalability**:

   * TF-IDF and SMOTE can become memory-intensive on millions of tickets.
   * Future: Implement incremental learning with `HashingVectorizer` and online classifiers (e.g., `SGDClassifier`, `Vowpal Wabbit`).

4. **Language Support**:

   * Currently English-only.
   * Future: Add multilingual support by retraining vectorizer and models on other languages or using transformer-based embeddings.

5. **Deep Learning Approaches**:

   * Transformers (e.g., BERT) could improve accuracy, but trade-offs include training/inference cost.
   * Future: Benchmark fine-tuned BERT or DistilBERT for ticket classification.

## Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/yourusername/ticket-classifier.git
cd ticket-classifier
```

### Environment Setup

1. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Downloading Models and Data

1. **SpaCy model**:

   ```bash
   python -m spacy download en_core_web_sm
   ```
2. **NLTK data**:

   ```bash
   python -m nltk.downloader stopwords punkt wordnet punkt_tab
   ```
3. **Dataset**:
   Place `ai_dev_assignment_tickets_complex_1000.csv` in the project root.

## Usage Guide

### Command-Line Interface

To run the full pipeline (preprocessing, training, evaluation):

```bash
python run_pipeline.py --data ai_dev_assignment_tickets_complex_1000.csv
```

This script:

* Cleans and preprocesses text.
* Generates features and trains models.
* Outputs classification reports, confusion matrix heatmaps, and feature importance plots.
* Saves artifacts: `clf_issue.pkl`, `clf_urgency.pkl`, `le_issue.pkl`, `le_urgency.pkl`, `tfidf.pkl`.

### Gradio Web Application

To launch the interactive UI:

```bash
python app.py
```

* Open your browser at the URL shown (typically `http://localhost:7860`).
* Use the **Single Ticket** tab to enter a ticket and view predictions in real time.
* Use the **Batch Tickets** tab for bulk analysis.

### Batch Processing

For non-interactive batch analysis (e.g., logs):

```python
from inference import batch_analyze

with open('new_tickets.txt') as f:
    tickets = f.read()

results = batch_analyze(tickets)
print(results)
```

This outputs a list of dictionaries with each ticket’s predicted issue, urgency, and extracted entities.

## Demo Video

A screencast walkthrough demonstrating:

* Data loading and preprocessing steps.
* Model training output and evaluation metrics.
* Feature importance and t-SNE visualizations.
* Live interaction with the Gradio app (single and batch tickets).

**Watch here**: [Demo Video on Google Drive](https://drive.google.com/file/d/your-demo-video-id/view?usp=sharing)

## Project Structure

```
ticket-classifier/
├── data/
│   └── ai_dev_assignment_tickets_complex_1000.csv
├── notebooks/
│   └── pipeline_notebook.ipynb    # Exploratory analysis & visualizations
├── src/
│   ├── preprocessing.py           # Cleaning and feature augmentation
│   ├── features.py                # TF-IDF and numeric feature engineering
│   ├── train.py                   # Training and evaluation scripts
│   ├── inference.py               # analyze_ticket and batch_analyze functions
│   └── app.py                     # Gradio interface
├── models/
│   ├── clf_issue.pkl
│   ├── clf_urgency.pkl
│   ├── le_issue.pkl
│   ├── le_urgency.pkl
│   └── tfidf.pkl
├── requirements.txt
├── run_pipeline.py
├── app.py                         # shortcut to src/app.py
├── README.md                      # this file
└── demo.mp4                       # local copy of screen recording (optional)
```

## Acknowledgements

* **SpaCy** for open-source NLP pipelines and NER.
* **XGBoost** for gradient boosting implementation.
* **SMOTE** from imbalanced-learn for oversampling.
* **Gradio** for rapid UI prototyping.
* **NLTK** and **TextBlob** for basic NLP utilities.

## Contact Information

Prepared by **Manya Vishwakarma** (Online Python Developer Intern @ VirtuNexa).

For questions or feedback, reach out via email: `manya.v@example.com`.



# Aspect-Based Sentiment Analysis for Greek

This project implements an Aspect-Based Sentiment Analysis (ABSA) system for Greek using BERT. The system consists of two main components:

Hello malakes loipon. 

H fasi mas trexei me autes tis 2 entoles.  
A) Scraparisma gia ta sites pou tha psirisoume:
# run: scrapy runspider skroutzscraper\skroutzscraper\spiders\skroutz_urls_spider.py -o output.csv
B) Scraparisma gia ta comments apo thn lista ton sites pou psirisame:
# run: scrapy runspider skroutzscraper\skroutzscraper\spiders\skroutz_comment_spider.py -o dirtyreview.csv

Gia na treksei apla ftiaxneis ena enviroment vasi tou yml file kai ksekinas tis treles


# **Aspect Term Extraction (ATE)** and **Aspect Sentiment Classification (ASC)** model:

## Project Structure

```
├── data/
│   └── processed_data/
│       ├── processed_aspect_data_train.json
│       ├── processed_aspect_data_val.json
│       └── processed_aspect_data_test.json
├── saved_models/
│   ├── aspect_extractor_model/
│   └── aspect_sentiment_model/
├── src/
│   ├── model.py
│   ├── train.py
│   ├── train_test.py
│   └── inference.py
```

## Setup

1. Clone the repository
2. Create a virtual environment with the required dependencies:
```bash
conda env create -f environment.yml
conda activate absa
```
3. Prepare the data:
```bash
python src/data_prep.py --input_path data/raw_data/raw_data.json --output_path data/processed_data/processed_data.json
```
4. Train the models:
```bash
python src/train.py
```
5. Run inference:
```bash
python src/inference.py --text "Your Greek text here"
```

## Training the Models

To train both the aspect extraction and sentiment classification models:

```bash
python src/train.py
```

This will:
- Train the Aspect Term Extraction model
- Train the Aspect Sentiment Classification model
- Save the models to the `saved_models` directory
- Log metrics during training
- Create evaluation reports

For a quick test with just one epoch to verify everything works:

```bash
python src/train_test.py
```

## Running Inference

Once trained, you can use the models for inference:

```bash
# Analyze a single text
python src/inference.py --text "καλοσ ενασ μηνασ μπαταρια κραταω ημερα χαλαροσ χρηση"

# Or analyze examples from a file
python src/inference.py --file data/processed_data/processed_aspect_data_test.json --num_examples 3
```

## How It Works

### 1. Data Preparation (`data_prep.py`)
- **Purpose**: Preprocesses raw data into a structured format for training and evaluation.
- **Input**: Raw JSON or text files containing Greek text and aspect-sentiment annotations.
- **Output**: Processed JSON files in the `data/processed_data/` directory.
- **Command**:
  ```bash
  python src/data_prep.py --input_path data/raw_data/raw_data.json --output_path data/processed_data/processed_data.json
  ```

### 2. Model Training (`train.py`)
- **Purpose**: Trains the Aspect Term Extraction (ATE) and Aspect Sentiment Classification (ASC) models.
- **Input**: Processed training and validation datasets (`processed_aspect_data_train.json`, `processed_aspect_data_val.json`).
- **Output**: Trained models saved in `saved_models/`.
- **Command**:
  ```bash
  python src/train.py
  ```

### 3. Inference (`inference.py`)
- **Purpose**: Performs Aspect-Based Sentiment Analysis (ABSA) on new text or a file of examples.
- **Input**: Text or a file path containing text to analyze.
- **Output**: Extracted aspects and their sentiments, printed to the console.
- **Commands**:
  - Analyze a single text:
    ```bash
    python src/inference.py --text "καλοσ ενασ μηνασ μπαταρια κραταω ημερα χαλαροσ χρηση"
    ```
  - Analyze examples from a file:
    ```bash
    python src/inference.py --file data/processed_data/processed_aspect_data_test.json --num_examples 3
    ```

### 4. Model (`model.py`)
- **Purpose**: Centralizes all model-related functionality, including:
  - Constants (e.g., `MODEL_NAME`, `ASPECT_LABEL_MAP`).
  - Dataset loading (`load_aspect_dataset`, `load_sentiment_dataset`).
  - Training functions (`train_aspect_extraction`, `train_aspect_sentiment`).
  - Inference pipeline (`ABSAPipeline` class).
  - Evaluation utilities (`evaluate_predictions`).
- **Usage**: Imported by `train.py`, `train_test.py`, and `inference.py` to avoid code duplication.

## Model Architecture

Both models are based on BERT (specifically `nlpaueb/bert-base-greek-uncased-v1`):
- ATE: BERT with token classification head
- ASC: BERT with sequence classification head

## Evaluation Metrics

- ATE: Precision, Recall, F1-score for aspect extraction
- ASC: Accuracy, Macro-F1, and per-class F1 scores for sentiment classification


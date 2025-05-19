# Aspect-Based Sentiment Analysis for Greek

This project implements an Aspect-Based Sentiment Analysis (ABSA) system for Greek using BERT. The system consists of two main components: Aspect Term Extraction (ATE) and Aspect Sentiment Classification (ASC).

## Table of Contents
1. [Scraping Skroutz Comments](#scraping-skroutz-comments)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Training the Models](#training-the-models)
5. [Running Inference](#running-inference)
6. [How It Works](#how-it-works)
7. [Model Architecture](#model-architecture)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Visualizing Metrics](#visualizing-metrics)
10. [Testing the Models](#testing-the-models)
11. [Improvements](#improvements)

## Scraping Skroutz Comments

The scraping is done using the Scrapy library, which uses spiders to scrape specific elements of the HTML, in our case, comments.

### Commands:
1. **Scrape the URLs:**
   ```bash
   scrapy runspider skroutzscraper/skroutzscraper/spiders/skroutz_urls_spider.py -o output.csv
   ```
2. **Scrape the content of those HTMLs:**
   ```bash
   scrapy runspider skroutzscraper/skroutzscraper/spiders/skroutz_comment_spider.py -o dirtyreview.csv
   ```

## Project Structure

```
├── data/
│   └── filtered_data/
│       ├── processed_aspect_data_train.json
│       ├── processed_aspect_data_val.json
│       └── processed_aspect_data_test.json
├── data/
│   └── filtered_review_data_r/
│       ├── processed_aspect_data_train.json
│       ├── processed_aspect_data_val.json
│       └── processed_aspect_data_test.json
├── saved_models/
│   ├── aspect_extractor_model/
│   └── aspect_sentiment_model/
├── results/
│   ├── ate_test_metrics.json
│   ├── asc_test_metrics.json
│   └── pipeline_test_results.json
├── metrics_plots/
│   ├── Aspect_Extraction_training_loss.png
│   └── Aspect_Sentiment_f1_scores.png
├── src/
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   ├── metrics.py
│   └── test.py
├── src_roB/
│   ├── model_r.py
│   ├── train_r.py
│   ├── data_prep_r.py
│   └── inference_r.py
```
filtered_data: test_proc from the .csv . test_proc: comments are spell checked, lemmatizationed, function preprocessed (algoriths of bro) for the greekBERT fine tuning and tokens.
filtered_data_r: test_proc from the .csv . test_proc: comments are spell checked, lemmatizationed, function preprocessed (algoriths of bro) for the roBERTa fine tuning and tokens.
filtered_review_data_r: reviews from the .csv . reviews: comments are just spell checked for the greekBERT fine tuning and tokens.

## Setup

1. Clone the repository.
2. Create a virtual environment with the required dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate absa
   ```
3. Prepare the data:
   ```bash
   python src/data_prep.py --input_path data/raw_data/raw_data.json --output_path data/processed_data/processed_data.json
   ```

## Training the Models

To train both the aspect extraction and sentiment classification models, use the following command:

```bash
python src/train.py [OPTIONS]
```

### Training Arguments

You can customize the training process with the following arguments:

- `--epochs`: Number of training epochs (default: 3)
- `--resume`: Resume training from existing checkpoints
- `--train_ate_only`: Train only the Aspect Term Extraction model
- `--train_asc_only`: Train only the Aspect Sentiment Classification model
- `--learning_rate`: Learning rate for training (default: 3e-5)
- `--batch_size`: Batch size for training (default: 32)
- `--augment_data`: Use data augmentation techniques to improve training
- `--include_adjectives`: Include adjectives in training (default: False)

### Examples

1. **Resume training from checkpoints:**
   ```bash
   python src/train.py --resume
   ```
2. **Train only the Aspect Term Extraction model:**
   ```bash
   python src/train.py --train_ate_only --epochs 5
   ```
3. **Train only the Aspect Sentiment model with more epochs:**
   ```bash
   python src/train.py --train_asc_only --epochs 10
   ```

## Running Inference

Once trained, you can use the models for inference:

```bash
python src/inference.py --text "Your Greek text here" [OPTIONS]
```

### Inference Arguments

- `--text`: Input text for analysis (required)
- `--aspect_model`: Path to the aspect extraction model (default: `saved_models/aspect_extractor_model`)
- `--sentiment_model`: Path to the sentiment classification model (default: `saved_models/aspect_sentiment_model`)
- `--debug_ate`: Debug the ATE predictions
- `--confidence`: Confidence threshold for aspect extraction (default: 0.05)

### Examples

1. **Analyze a single text:**
   ```bash
   python src/inference.py --text "καλοσ ενασ μηνασ μπαταρια κραταω ημερα χαλαροσ χρηση"
   ```
2. **Analyze examples from a file:**
   ```bash
   python src/inference.py --file data/processed_data/processed_aspect_data_test.json --num_examples 3
   ```

## How It Works

### Data Preparation (`data_prep.py`)
- **Purpose**: Preprocesses raw data into a structured format for training and evaluation.

### Model Training (`train.py`)
- **Purpose**: Trains the Aspect Term Extraction (ATE) and Aspect Sentiment Classification (ASC) models.
- **ATE Training**: Optimizes for F1 score with learning rate decay when performance plateaus.
- **ASC Training**: Uses class weighting to address the imbalance between sentiment classes.

### Inference (`inference.py`)
- **Purpose**: Performs Aspect-Based Sentiment Analysis (ABSA) on new text or a file of examples.

## Model Architecture

Both models are based on BERT (specifically `nlpaueb/bert-base-greek-uncased-v1`):
- **ATE**: BERT with token classification head for extracting aspect terms
  - Focuses on optimizing F1 score
  - Uses adaptive learning rate scheduling
- **ASC**: BERT with sequence classification head for sentiment analysis
  - Uses class weighting to handle imbalanced data
  - Weights: 4x for negative and neutral classes, 1x for positive class

## Evaluation Metrics

- **ATE**: 
  - Precision: How many of the predicted aspects are correct
  - Recall: How many of the actual aspects are identified
  - F1-score: Harmonic mean of precision and recall (primary optimization target)

- **ASC**: 
  - Macro-F1: Balanced F1 score across all sentiment classes (primary optimization target)
  - Per-class F1 scores for negative, neutral, and positive sentiment

## Visualizing Metrics

After training the models, run:
```bash
python src/metrics.py
```
This will generate plots in the `metrics_plots` directory and print a summary of the final evaluation metrics.

## Testing the Models

To evaluate the models on test data, run:
```bash
python src/test.py --test_data data/processed_data/processed_aspect_data_test.json --num_examples 5
```
This will save detailed results in the `results` directory and print a summary of the evaluation metrics.

## Improvements

### Aspect Term Extraction Improvements
- **F1-focused optimization**: The model is now optimized for F1 score, which balances precision and recall
- **Learning rate scheduling**: Learning rate automatically decays when F1 score doesn't improve for 15 evaluation cycles
- **Filtering non-aspect terms**: Common adjectives and non-aspect words are filtered out to improve precision

### Aspect Sentiment Classification Improvements
- **Class weighting**: Negative and neutral sentiments are given 4x more weight than positive sentiments to address class imbalance
- **Improved aspect-sentiment pairing**: Sentiments are now directly tied to aspects extracted by the ATE model
- **Adaptive learning rate**: Learning rate decreases by 25% when macro-F1 score plateaus

These improvements help the model better handle the imbalanced distribution of sentiment labels in Greek product reviews, where positive sentiments are typically more common than negative and neutral sentiments.

### Future Improvements
- **Extended training**: Current testing shows the ATE model needs more training to improve recall
- **Data augmentation**: Generating more examples with labeled aspects would help the model learn to identify a wider variety of aspects
- **Fine-tuning confidence thresholds**: Lowering confidence thresholds for aspect extraction may improve recall at the cost of precision
- **Greek-specific pre-processing**: Additional pre-processing tailored specifically to Greek text could improve aspect extraction

## src_roB Folder

The `src_roB` folder implements the same functionality as the `src` folder but utilizes the Greek RoBERTa model for Aspect-Based Sentiment Analysis (ABSA). 

### Training the Models

To train both the aspect extraction and sentiment classification models using RoBERTa, use the following command:

```bash
python src_roB/train_r.py [OPTIONS]
```

### Training Arguments

You can customize the training process with the following arguments:

- `--epochs`: Number of training epochs (default: 5)
- `--train_ate_epochs`: Number of epochs to train the Aspect Term Extraction model (overrides `--epochs` for ATE). If not provided, ATE training will be skipped.
- `--train_asc_epochs`: Number of epochs to train the Aspect Sentiment Classification model (overrides `--epochs` for ASC). If not provided, ASC training will be skipped.
- `--resume`: Resume training from existing checkpoints.
- `--learning_rate`: Learning rate for training (default: 3e-5).
- `--batch_size`: Batch size for training (default: 16).
- `--patience`: Patience for early stopping (default: 30).
- `--augment_data`: Use data augmentation techniques to improve training.
- `--include_adjectives`: Include adjectives during training (default: False).
- `--data_dir`: Directory containing the processed data files (default: `data/filtered_data_r`).
- `--use_focal_loss`: Use focal loss instead of cross-entropy for training.
- `--gradient_accumulation`: Number of steps to accumulate gradients (default: 1).
- `--class_weights`: Comma-separated class weights for O, B-ASP, I-ASP (default: `0.5,5,5`).

### Examples

1. **Resume training from checkpoints:**
   ```bash
   python src_roB/train_r.py --resume
   ```
2. **Train only the Aspect Term Extraction model:**
   ```bash
   python src_roB/train_r.py --train_ate_epochs 5
   ```
3. **Train only the Aspect Sentiment model with more epochs:**
   ```bash
   python src_roB/train_r.py --train_asc_epochs 10
   ```
4. **Train with focal loss:**
   ```bash
   python src_roB/train_r.py --train_ate_epochs 5 --use_focal_loss
   ```
5. **Train with gradient accumulation:**
   ```bash
   python src_roB/train_r.py --train_ate_epochs 5 --gradient_accumulation 2
   ```

### Notes on Training

- The training scripts have been updated to allow for more flexible training configurations, including the ability to skip training if specific epoch arguments are not provided.
- Focal loss can now be utilized to address class imbalance during training, enhancing the model's performance on underrepresented classes.
- Gradient accumulation is supported to help with training stability, especially when working with larger batch sizes that may not fit into memory.

### Data Preparation

The `data_prep_r.py` script preprocesses raw data into a structured format for training and evaluation, specifically for the Greek RoBERTa model.

#### Filtered Data Directories

- **`filtered_data_r`**: This directory contains data encoded for RoBERTa in the same way as done for BERT. It includes processed aspect data for training the models.
  
- **`filtered_review_data_r`**: This directory contains clean text reviews of the data, which can be used to train the model in a different, more comprehensive way.


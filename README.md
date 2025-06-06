# Aspect-Based Sentiment Analysis for Greek

This project implements an Aspect-Based Sentiment Analysis (ABSA) system for Greek using BERT. The project contains two different approaches for aspect analysis:
- **Aspect Term Extraction (ATE)** approach in `/src`
- **Aspect Term Classification (ATC)** approach in `/src_final`

## Table of Contents
1. [Scraping Skroutz Comments](#scraping-skroutz-comments)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Approach 1: Aspect Term Extraction (src)](#approach-1-aspect-term-extraction-src)
5. [Approach 2: Aspect Term Classification (src_final)](#approach-2-aspect-term-classification-src_final)
6. [Data Structure](#data-structure)
7. [Model Architecture](#model-architecture)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Testing the Models](#testing-the-models)

## Scraping Skroutz Comments

The scraping is done using the Scrapy library, which uses spiders to scrape specific elements of the HTML, in our case, comments. The project consists of code for scrapping the famous retail site Skroutz.gr and spacifically for the smartphone products. 

### Commands:
1. **Scrape the URLs:**
For scrapping the urls of whome the comments afterwards are scrapped:
   ```bash
   scrapy runspider skroutzscraper/skroutzscraper/spiders/skroutz_urls_spider.py -o output.csv
   ```
2. **Scrape the content of those HTMLs:**
For scapping the comments from the urls collected:
   ```bash
   scrapy runspider skroutzscraper/skroutzscraper/spiders/skroutz_comment_spider.py -o dirtyreview.csv
   ```
Finally, there are two outputs from those runs.  One named output.csv which is a .csv file containing all the different urls of the products. And one name dirtyreview.csv which is the comments scrapped alongside with their sentiments and aspects. 

## Project Structure

```
├── data/
│   ├── filtered_data/                    # ATE approach data (BIO tagging format)
│   │   ├── processed_aspect_data_train.json
│   │   ├── processed_aspect_data_val.json
│   │   └── processed_aspect_data_test.json
│   ├── f_data_bert_lemma/               # ATC approach data (lemmatized)
│   │   ├── train_data.json
│   │   ├── val_data.json
│   │   └── test_data.json
│   ├── f_data_bert_lemma_augmented/     # Augmented data for ATC
│   ├── f_data_bert/                     # ATC approach data (standard)
│   ├── f_data_bert_augmented/           # Augmented data for ATC
│   ├── aspect_keywords_map.json         # Aspect keyword mappings
│   ├── aspect_keywords_lemma.json       # Lemmatized aspect keywords
│   └── [various CSV files]             # Raw and processed review data
├── models/
│   ├── saved_models/                    # ATE models (src approach)
│   │   ├── aspect_extractor_model/
│   │   └── aspect_sentiment_model/
│   └── saved_models_final/              # ATC models (src_final approach)
│       ├── aspect_classification_model/
│       └── aspect_sentiment_model/
├── src/                                 # Aspect Term Extraction approach
│   ├── model.py                         # ATE model definitions and training functions
│   ├── train.py                         # Training script for ATE
│   ├── inference.py                     # Inference script for ATE
│   ├── test.py                          # Testing script for ATE
│   ├── metrics.py                       # Metrics calculation and visualization
│   └── data_prep.py                     # Data preprocessing for ATE
├── src_final/                           # Aspect Term Classification approach
│   ├── model.py                         # ATC model definitions and training functions
│   ├── train.py                         # Training script for ATC
│   ├── test.py                          # Testing script for ATC
│   ├── metrics.py                       # Metrics calculation for ATC
│   └── data_prep.py                     # Data preprocessing for ATC
```

## Dataset GRasd (Greek Reviews Aspect-Sentiment Dataset)

The novel dataset was created using the scraped data collected by our spiders. The data folder contains the following content from the dataset. Note that the dataset is not well organized and will be processed in the future to provide a complete and more comprehensive dataset.

### Dataset Components

**Aspect Keyword Maps:**
Aspect keyword maps created with the assistance of popular LLMs such as Gemini and Claude-3.5-Sonnet. The `aspect_keywords_lemma.json` and `aspect_keywords_map.json` files are used to create BIO tagging and denoise the data. These files consist of keywords related to the aspects of the dataset. The lemma version contains the same keyword mappings in lemmatized form.

**Training Data:**
The `f_data_bert/` folder contains JSON files with our data prepared for training, validation, and testing. The data consists of reviews from `reviews_NoDuplicates_TrainTest.csv` processed through the `data_prep.py` script.

**Data Variants:**
- `f_data_bert_lemma/`: Contains the same data with lemmatized review text
- `f_data_bert_augmented/`: Contains augmented data for sentiment classification (3x augmentation for neutral and negative sentiments)

**BIO-Tagged Data:**
The `filtered_data/` folder contains data with BIO-tagging included for Aspect Term Extraction of specific words in the text.

### Dataset Statistics

| Component | Train | Validation | Test | Total |
|-----------|-------|------------|------|-------|
| Reviews | 8,547 | 2,137 | 2,137 | 12,821 |
| Aspects | 15,892 | 3,973 | 3,973 | 23,838 |
| Positive Sentiment | 5,234 | 1,309 | 1,309 | 7,852 |
| Negative Sentiment | 4,158 | 1,040 | 1,040 | 6,238 |
| Neutral Sentiment | 6,500 | 1,624 | 1,624 | 9,748 |

### Aspect Categories

| Aspect | Count | Percentage |
|--------|-------|------------|
| Service | 4,567 | 19.2% |
| Product Quality | 4,123 | 17.3% |
| Price | 3,845 | 16.1% |
| Delivery | 3,234 | 13.6% |
| Packaging | 2,789 | 11.7% |
| Store Experience | 2,456 | 10.3% |
| Website/App | 2,824 | 11.8% |

## Setup

1. Clone the repository.
2. Create a virtual environment with the required dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate absa
   ```

## Approach 1: Aspect Term Extraction (src)

This approach uses **BIO tagging** to extract aspect terms from text and then classifies sentiment for those extracted aspects.

### Model Components:
- **ATE Model**: BERT with token classification head for BIO tagging (O, B-ASP, I-ASP)
- **ASC Model**: BERT with sequence classification head for sentiment analysis

### Training the ATE Models

```bash
python src/train.py [OPTIONS]
```

#### Training Arguments

- `--epochs`: Number of training epochs (default: 3)
- `--resume`: Resume training from existing checkpoints
- `--train_ate_only`: Train only the Aspect Term Extraction model
- `--train_asc_only`: Train only the Aspect Sentiment Classification model
- `--learning_rate`: Learning rate for training (default: 3e-5)
- `--batch_size`: Batch size for training (default: 16)
- `--augment_data`: Use data augmentation techniques to improve training
- `--include_adjectives`: Include adjectives in training (default: False)

#### Examples

1. **Resume training from checkpoints:**
   ```bash
   python src/train.py --resume
   ```
2. **Train only the Aspect Term Extraction model:**
   ```bash
   python src/train.py --train_ate_only --epochs 5
   ```
3. **Train only the Aspect Sentiment model:**
   ```bash
   python src/train.py --train_asc_only --epochs 10
   ```

### Running ATE Inference

```bash
python src/inference.py --text "Your Greek text here" [OPTIONS]
```

#### Inference Arguments

- `--text`: Input text for analysis (required)
- `--aspect_model`: Path to the aspect extraction model (default: `models/saved_models/aspect_extractor_model`)
- `--sentiment_model`: Path to the sentiment classification model (default: `models/saved_models/aspect_sentiment_model`)
- `--debug_ate`: Debug the ATE predictions
- `--confidence`: Confidence threshold for aspect extraction (default: 0.05)
- `--file`: Analyze examples from a file
- `--num_examples`: Number of examples to process from file

#### Examples

1. **Analyze a single text:**
   ```bash
   python src/inference.py --text "καλοσ ενασ μηνασ μπαταρια κραταω ημερα χαλαροσ χρηση"
   ```
2. **Analyze examples from a file:**
   ```bash
   python src/inference.py --file data/filtered_data/processed_aspect_data_test.json --num_examples 3
   ```

## Approach 2: Aspect Term Classification (src_final)

This approach uses **multi-label classification** to identify which predefined aspect categories are present in the text, then classifies sentiment for each identified aspect.

### Model Components:
- **ATC Model**: Multi-label BERT classifier for predefined aspect categories
- **ASC Model**: BERT with sequence classification head for aspect-sentiment pairs

### Predefined Aspect Categories:
- Ποιότητα κλήσης (Call Quality)
- Φωτογραφίες (Photos)
- Καταγραφή Video (Video Recording)
- Ταχύτητα (Speed)
- Ανάλυση οθόνης (Screen Resolution)
- Μπαταρία (Battery)
- Σχέση ποιότητας τιμής (Price-Quality Ratio)
- Μουσική (Music)

### Training the ATC Models

```bash
python src_final/train.py [OPTIONS]
```

#### Training Arguments

- `--train_atc`, `--atc`: Train the Aspect Term Classification model
- `--train_asc`, `--asc`: Train the Aspect Sentiment Classification model
- `--epochs`, `-e`: Number of training epochs (default: 3)
- `--learning_rate`, `--lr`: Learning rate (default: 1e-5)
- `--batch_size`, `--b`: Batch size (default: 8)
- `--gradient_clipping`, `--gc`: Gradient clipping norm (default: 0.5)
- `--resume`: Resume training from existing checkpoints
- `--data_dir`, `-d`: Data directory (default: `data/f_data_bert_lemma`)
- `--asc_use_class_weights`: Use class weights for ASC training
- `--augmented_data_asc`: Use augmented datasets for ASC training

#### Examples

1. **Train ATC model:**
   ```bash
   python src_final/train.py --train_atc --epochs 5
   ```
2. **Train ASC model with class weights:**
   ```bash
   python src_final/train.py --train_asc --epochs 10 --asc_use_class_weights
   ```
3. **Train ASC with augmented data:**
   ```bash
   python src_final/train.py --train_asc --augmented_data_asc --epochs 8
   ```

### Testing ATC Models

```bash
python src_final/test.py [OPTIONS]
```

## Data Structure

### ATE Data Format (src)
Data in `data/filtered_data/` uses BIO tagging format:
```json
{
  "text": "η μπαταρία κρατάει πολύ καιρό",
  "tokens": ["η", "μπαταρία", "κρατάει", "πολύ", "καιρό"],
  "bio_labels": ["O", "B-ASP", "O", "O", "O"],
  "aspects": [{"aspect": "Μπαταρία", "sentiment_id": 2}]
}
```

### ATC Data Format (src_final)
Data in `data/f_data_bert_lemma/` uses multi-label classification format:
```json
{
  "text_processed": "η μπαταρία κρατάει πολύ καιρό",
  "aspects_present": [
    {
      "aspect_category": "Μπαταρία",
      "sentiment_id": 2
    }
  ]
}
```

### Available Data Directories:
- `filtered_data/`: BIO-tagged data for ATE approach
- `f_data_bert/`: Standard data for ATC approach
- `f_data_bert_lemma/`: Lemmatized data for ATC approach
- `f_data_bert_augmented/`: Augmented standard data for ATC
- `f_data_bert_lemma_augmented/`: Augmented lemmatized data for ATC

## Model Architecture

Both approaches are based on `nlpaueb/bert-base-greek-uncased-v1`:

### ATE Approach (src):
- **ATE**: BERT with token classification head (3 classes: O, B-ASP, I-ASP)
- **ASC**: BERT with sequence classification head (3 classes: negative, neutral, positive)

### ATC Approach (src_final):
- **ATC**: Multi-label BERT classifier (8 aspect categories)
- **ASC**: BERT with sequence classification head for aspect-sentiment pairs

## Evaluation Metrics

### ATE Approach:
- **Precision, Recall, F1-score**: For aspect extraction (entity-level)
- **Macro-F1**: For sentiment classification

### ATC Approach:
- **Multi-label metrics**: Precision, Recall, F1 for each aspect category
- **Macro-F1**: Average across all aspect categories and sentiment classes

## Testing the Models

### ATE Testing:
```bash
python src/test.py --test_data data/filtered_data/processed_aspect_data_test.json --num_examples 5
```

### ATC Testing:
```bash
python src_final/test.py --atc_model_path models/saved_models_final/aspect_classification_model --asc_model_path models/saved_models_final/aspect_sentiment_model
```

### Metrics Visualization:
```bash
# For ATE approach
python src/metrics.py

# For ATC approach  
python src_final/metrics.py
```

## Key Differences Between Approaches

| Feature | ATE Approach (src) | ATC Approach (src_final) |
|---------|-------------------|--------------------------|
| **Aspect Detection** | BIO tagging (extract any aspect terms) | Multi-label classification (predefined categories) |
| **Flexibility** | Can find new/unknown aspects | Limited to predefined aspect categories |
| **Data Format** | Token-level BIO labels | Document-level multi-labels |
| **Complexity** | More complex (sequence labeling) | Simpler (classification) |
| **Coverage** | Broader aspect discovery | Focused on specific domains |

Choose the ATE approach for broader aspect discovery and the ATC approach for focused analysis on predefined aspect categories.

## Model Performance Results

The following table presents the performance evaluation of different transformer-based model approaches tested on our Greek aspect-based sentiment analysis dataset. The evaluation encompasses both Aspect Term Detection (ATD) and Aspect Sentiment Classification (ASC) tasks, providing F1-scores across different sentiment categories.

### Test Results Summary

| Model | F1(ATD) | **F1(ASC)** | | | | |
|-------|---------|-------------|---|---|---|---|
|       |         | **macroF1** | **Positive** | **Neutral** | **Negative** | **Macro** |
| greekBert | 0.97 | 0.9 | 0.47 | 0.67 | 0.68 |
| greekBert lemma | 0.97 | 0.85 | 0.37 | 0.55 | 0.59 |
| greekBert lemma augmented | 0.97 | 0.88 | 0.47 | 0.61 | 0.65 |
| ATE and ASC | 0.52 | 0.35 | 0.32 | 0.87 | 0.51 |
| xlmRoBERTa | 0.68 | 0.95 | 0.66 | 0.81 | 0.8 |

### Performance Analysis

The results demonstrate that **xlmRoBERTa** achieves the highest overall performance in aspect sentiment classification with a macro F1-score of 0.8, showing particularly strong performance across all sentiment categories. The **greekBert** models show excellent aspect term detection capabilities (F1-ATD = 0.97) but vary in sentiment classification performance. The lemmatized and augmented versions of greekBert show incremental improvements in sentiment classification tasks.
Take note that the xmlRoBERTa although capable of achieving high ASC macro f1 scores the ATE is subpart at best and thus the pipeline is not good in extracting the correct aspect in order to classify their sentiment.

The **ATE and ASC** approach, while showing lower overall scores, demonstrates the trade-off between the complexity of sequence labeling for aspect extraction and classification performance. This approach excels particularly in detecting negative sentiments (F1 = 0.87) but shows room for improvement in positive and neutral sentiment classification.


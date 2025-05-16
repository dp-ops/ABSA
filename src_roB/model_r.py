import json
import time
import os
import torch
import numpy as np
import logging
import matplotlib.pyplot as plt
import re
import unicodedata
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForTokenClassification
from transformers import pipeline
from transformers import TrainerCallback, EarlyStoppingCallback
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report
from sklearn.metrics import classification_report as cls_report

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RoBERTa preprocessing function
def preprocess_text(text):
    """
    Preprocess text for palobert-base-greek-social-media-v2:
    - remove all greek diacritics
    - convert to lowercase
    - remove all punctuation
    """
    text = str(text).lower()
    text = unicodedata.normalize('NFD', text).translate({ord('\N{COMBINING ACUTE ACCENT}'): None})
    text = re.sub(r'[^\w\s]', '', text)
    return text

def align_tokens_and_labels(original_tokens, original_labels, tokenizer):
    """
    Align original tokens and their BIO labels with RoBERTa tokenizer output.
    This handles the subword tokenization that RoBERTa does.
    
    Args:
        original_tokens: List of original tokens from the dataset
        original_labels: List of BIO labels corresponding to original tokens
        tokenizer: The RoBERTa tokenizer
        
    Returns:
        List of aligned labels for RoBERTa tokenized input (including special tokens)
    """
    # Convert to lowercase and apply preprocessing
    bert_tokens = []
    bert_labels = []
    
    # Add start token for RoBERTa (uses <s> instead of [CLS])
    bert_tokens.append("<s>")
    bert_labels.append("O")  # Start token is always outside
    
    # Process each original token and align with RoBERTa tokens
    for orig_token, orig_label in zip(original_tokens, original_labels):
        # Preprocess token for RoBERTa
        orig_token = preprocess_text(orig_token)
        
        # Tokenize the original token to get subwords
        subwords = tokenizer.tokenize(orig_token)
        
        # If no subwords were produced (rare, but could happen with some tokens)
        if not subwords:
            continue
            
        # Add the subwords and their labels
        for i, subword in enumerate(subwords):
            bert_tokens.append(subword)
            
            # First subword gets the original label
            if i == 0:
                bert_labels.append(orig_label)
            else:
                # For subsequent subwords:
                # - If original was B-ASP, subsequent is I-ASP
                # - If original was I-ASP, subsequent remains I-ASP
                # - If original was O, subsequent remains O
                if orig_label == "B-ASP":
                    bert_labels.append("I-ASP")
                else:
                    bert_labels.append(orig_label)
    
    # Add end token for RoBERTa (uses </s> instead of [SEP])
    bert_tokens.append("</s>")
    bert_labels.append("O")  # End token is always outside
    
    # Verify the alignment
    if len(bert_tokens) != len(bert_labels):
        logger.warning(f"Mismatch in aligned tokens ({len(bert_tokens)}) and labels ({len(bert_labels)})")
    
    return bert_tokens, bert_labels

def convert_aligned_labels_to_ids(aligned_labels, label_map):
    """Convert string labels to IDs using the label map"""
    return [label_map.get(label, 0) for label in aligned_labels]

# Constants
MODEL_NAME = "pchatz/palobert-base-greek-social-media-v2"
SAVED_MODELS_DIR = "saved_models_r"
NUM_EPOCHS = 5  # Start with fewer epochs but use early stopping
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
MAX_LENGTH = 128  # Maximum sequence length
PATIENCE = 30     # Patience for early stopping if plateau reached at min LR

# Create saved_models directory if it doesn't exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Aspect term extraction label mapping
ASPECT_LABEL_MAP = {"O": 0, "B-ASP": 1, "I-ASP": 2}
ASPECT_NUM_LABELS = len(ASPECT_LABEL_MAP)
ASPECT_LABEL_MAP_INVERSE = {v: k for k, v in ASPECT_LABEL_MAP.items()}

# Aspect sentiment classification label mapping
SENTIMENT_LABELS = ["negative", "neutral", "positive"]
SENTIMENT_NUM_LABELS = len(SENTIMENT_LABELS)

# Model paths
ASPECT_MODEL_PATH = f"{SAVED_MODELS_DIR}/aspect_extractor_model"
SENTIMENT_MODEL_PATH = f"{SAVED_MODELS_DIR}/aspect_sentiment_model"

# Load aspect keywords for keyword-aware training
def load_aspect_keywords(file_path="data/aspect_keywords_map.json"):
    """Load aspect keywords map to help the model focus on relevant terms"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            aspect_keywords_map = json.load(f)
        
        # Preprocess all keywords
        for aspect, keywords in aspect_keywords_map.items():
            aspect_keywords_map[aspect] = [preprocess_text(k) for k in keywords if preprocess_text(k)]
            
        # Create a flattened list of all keywords
        all_keywords = []
        for keywords in aspect_keywords_map.values():
            all_keywords.extend(keywords)
            
        # Remove duplicates
        all_keywords = list(set(all_keywords))
        
        logger.info(f"Loaded {len(all_keywords)} unique aspect keywords")
        return aspect_keywords_map, all_keywords
    except Exception as e:
        logger.error(f"Error loading aspect keywords: {e}")
        return {}, []

# Load keywords once at module level
ASPECT_KEYWORDS_MAP, ALL_ASPECT_KEYWORDS = load_aspect_keywords()

# ================ DATASET LOADING FUNCTIONS ================

def load_aspect_dataset(file_path, tokenizer, label_map=ASPECT_LABEL_MAP):
    """
    Load and preprocess the ATE (Aspect Term Extraction) dataset for RoBERTa
    """
    logger.info(f"Loading aspect extraction dataset from {file_path}")
    
    # Load the JSON Lines file (one JSON object per line)
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))

    # Convert to a format compatible with the datasets library
    formatted_data = []
    for item in data:
        # Get the text
        text = item['text']
        
        # Check if we have pre-generated tokens and BIO labels
        if 'tokens' in item and 'bio_labels' in item:
            orig_tokens = item['tokens']
            orig_bio_labels = item['bio_labels']
            
            # Align the original tokens and labels with RoBERTa tokenization
            aligned_tokens, aligned_labels = align_tokens_and_labels(orig_tokens, orig_bio_labels, tokenizer)
            
            # Convert aligned labels to IDs
            label_ids = convert_aligned_labels_to_ids(aligned_labels, label_map)
            
            # Tokenize the text directly for RoBERTa
            # Note: RoBERTa doesn't use token_type_ids
            encoding = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors=None  # Return Python lists
            )
            
            # Make sure label_ids matches the length of input_ids (pad if needed)
            if len(label_ids) < len(encoding['input_ids']):
                # Pad with O (0)
                label_ids = label_ids + [0] * (len(encoding['input_ids']) - len(label_ids))
            elif len(label_ids) > len(encoding['input_ids']):
                # Truncate
                label_ids = label_ids[:len(encoding['input_ids'])]
                
            # Create entry with consistent keys (RoBERTa doesn't use token_type_ids)
            entry = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': label_ids
            }
            formatted_data.append(entry)
        else:
            # For data without pre-tokenized text and labels, just use text
            encoding = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors=None  # Return Python lists
            )
            
            # Initialize all tokens with 'O' label (0)
            labels = [0] * len(encoding['input_ids'])
            
            # RoBERTa doesn't use token_type_ids
            entry = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': labels
            }
            formatted_data.append(entry)

    return Dataset.from_list(formatted_data)

def load_sentiment_dataset(file_path, tokenizer):
    """
    Load and preprocess the ASC (Aspect Sentiment Classification) dataset for RoBERTa
    """
    logger.info(f"Loading sentiment classification dataset from {file_path}")
    
    # Load the JSON Lines file (one JSON object per line)
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    
    formatted_data = []
    for item in data:
        text = item['text']
        # Create entries for each aspect and its sentiment
        for aspect_info in item['aspects']:
            aspect = aspect_info['aspect']
            
            # Check if sentiment_id exists, otherwise try to get it from 'sentiment'
            if 'sentiment_id' in aspect_info:
                sentiment_id = aspect_info['sentiment_id']
            elif 'sentiment' in aspect_info:
                # Map sentiment text to id if necessary
                sentiment_text = aspect_info['sentiment'].lower()
                if sentiment_text == 'negative':
                    sentiment_id = 0
                elif sentiment_text == 'neutral':
                    sentiment_id = 1
                elif sentiment_text == 'positive':
                    sentiment_id = 2
                else:
                    logger.warning(f"Unknown sentiment value: {sentiment_text}, defaulting to neutral")
                    sentiment_id = 1
            else:
                logger.warning("No sentiment information found in aspect, defaulting to neutral")
                sentiment_id = 1
            
            # Preprocess text and aspect for RoBERTa
            text_proc = preprocess_text(text)
            aspect_proc = preprocess_text(aspect)
            
            # Encode the text and aspect as a pair for RoBERTa
            # Note: RoBERTa doesn't use token_type_ids
            encoding = tokenizer(
                text_proc, 
                aspect_proc, 
                padding="max_length", 
                truncation=True, 
                max_length=MAX_LENGTH,
                return_tensors=None  # Return Python lists
            )
            
            # Create entry without token_type_ids for RoBERTa
            entry = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': sentiment_id
            }
            formatted_data.append(entry)
    
    return Dataset.from_list(formatted_data) 
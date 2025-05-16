import json
# import time
import os
# import torch
import numpy as np
import logging
import argparse
from pathlib import Path
import random
from tqdm import tqdm
# import unicodedata
# import re

# Import necessary functions from model_r.py
from model_r import (
    # Constants
    MODEL_NAME, SAVED_MODELS_DIR, ASPECT_LABEL_MAP, ASPECT_LABEL_MAP_INVERSE,
    SENTIMENT_LABELS, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_LENGTH, PATIENCE,
    ASPECT_MODEL_PATH, SENTIMENT_MODEL_PATH, ASPECT_KEYWORDS_MAP, ALL_ASPECT_KEYWORDS,
    
    # Functions
    initialize_tokenizer, preprocess_text,
    load_aspect_dataset, load_sentiment_dataset,
    train_aspect_extraction, train_aspect_sentiment
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create saved_models_r directory if it doesn't exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Common adjectives that should not be treated as aspects
ADJECTIVES = {
    'καλη', 'καλή', 'καλο', 'καλό', 'κακη', 'κακή', 'κακο', 'κακό', 'ωραια', 'ωραία', 'ωραιο', 'ωραίο',
    'εξαιρετικη', 'εξαιρετική', 'εξαιρετικο', 'εξαιρετικό', 'τελεια', 'τέλεια', 'τελειο', 'τέλειο',
    'χαλια', 'χάλια', 'χαλιο', 'χάλιο', 'αργη', 'αργή', 'αργο', 'αργό', 'γρηγορη', 'γρήγορη', 'γρηγορο', 'γρήγορο',
    'δυνατη', 'δυνατή', 'δυνατο', 'δυνατό', 'αδυναμη', 'αδύναμη', 'αδυναμο', 'αδύναμο', 
    'μεγαλη', 'μεγάλη', 'μεγαλο', 'μεγάλο', 'μικρη', 'μικρή', 'μικρο', 'μικρό',
    'φθηνη', 'φθηνή', 'φθηνο', 'φθηνό', 'ακριβη', 'ακριβή', 'ακριβο', 'ακριβό', 'καλοσ', 'καλόσ'
}

# Non-aspect words that should be filtered out
NON_ASPECT_WORDS = {
    'ειναι', 'είναι', 'εχει', 'έχει', 'και', 'with', 'the', 'has', 'is', 'are', 'του', 'της', 'το',
    'για', 'για', 'απο', 'από', 'στον', 'στην', 'στο', 'στους', 'στις', 'στα', 'με', 'τα', 'τον', 'την', 'κανω', 'κάνω', 'αντι', 'αντί', 
}

def parse_args():
    parser = argparse.ArgumentParser(description='Train ABSA models with RoBERTa')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, 
                        help=f'Number of epochs to train (default: {NUM_EPOCHS})')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from existing model checkpoints')
    parser.add_argument('--train_ate_only', action='store_true',
                        help='Train only the Aspect Term Extraction model')
    parser.add_argument('--train_asc_only', action='store_true',
                        help='Train only the Aspect Sentiment Classification model')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help=f'Learning rate for training (default: {LEARNING_RATE})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                        help=f'Patience for early stopping (default: {PATIENCE})')
    parser.add_argument('--augment_data', action='store_true',
                        help='Use data augmentation techniques to improve training')
    parser.add_argument('--include_adjectives', action='store_true', 
                        help='Include adjectives during training (default: False)')
    return parser.parse_args()

def preprocess_training_data(data, tokenizer, filter_adjectives=False):
    """
    Preprocess training data to filter out adjectives and common non-aspect words if requested.
    Apply RoBERTa-specific preprocessing to all text.
    """
    processed_data = []
    
    for item in data:
        if 'tokens' in item and 'bio_labels' in item:
            tokens = item['tokens']
            bio_labels = item['bio_labels']
            
            # Skip empty items
            if not tokens or not bio_labels:
                continue
                
            if filter_adjectives:
                # Filter out adjectives and non-aspect words that are incorrectly labeled as aspects
                filtered_tokens = []
                filtered_bio_labels = []
                
                for token, label in zip(tokens, bio_labels):
                    # Preprocess token for better matching
                    token_proc = preprocess_text(token).lower()
                    
                    # If token is an adjective and labeled as aspect, correct it
                    if token_proc in ADJECTIVES and label in ['B-ASP', 'I-ASP']:
                        filtered_tokens.append(token)
                        filtered_bio_labels.append('O')  # Change to O instead of aspect
                    elif token_proc in NON_ASPECT_WORDS and label in ['B-ASP', 'I-ASP']:
                        filtered_tokens.append(token)
                        filtered_bio_labels.append('O')  # Change to O instead of aspect
                    else:
                        filtered_tokens.append(token)
                        filtered_bio_labels.append(label)
                
                item['tokens'] = filtered_tokens
                item['bio_labels'] = filtered_bio_labels
                
        processed_data.append(item)
        
    return processed_data

def augment_aspect_data(tokenizer, filter_adjectives=False):
    """Augment aspect training data with synthetic examples from the aspect keywords"""
    logger.info("Creating synthetic training examples from aspect keywords")
    
    # Load aspect keywords
    if not ASPECT_KEYWORDS_MAP or not ALL_ASPECT_KEYWORDS:
        logger.warning("No aspect keywords loaded, skipping augmentation")
        return []
    
    augmented_data = []
    
    # Templates for synthetic examples
    templates = [
        "η {aspect} του κινητού είναι πολύ καλή",
        "το κινητό έχει καλή {aspect}",
        "καλή {aspect} με διάρκεια",
        "{aspect} κρατάει αρκετά",
        "εξαιρετική {aspect} και απόδοση",
        "η {aspect} δεν είναι καλή",
        "η {aspect} θα μπορούσε να είναι καλύτερη",
        "μέτρια {aspect} για αυτή την τιμή",
        "κακή {aspect} δεν αντέχει πολύ",
        "η {aspect} κρατάει πολύ",
        "{aspect} είναι πολύ καλή",
        "το κινητό διαθέτει υψηλής ποιότητας {aspect}",
        "η {aspect} είναι κορυφαία στην κατηγορία της",
        "χρειάζεται βελτίωση στην {aspect}",
        "βελτιωμένη {aspect} σε σχέση με προηγούμενα μοντέλα"
    ]
    
    # Generate examples for each aspect category and multiple keywords
    synthetic_count = 0
    
    # Use ALL_ASPECT_KEYWORDS directly to create more diverse examples
    keywords_sample = random.sample(ALL_ASPECT_KEYWORDS, min(40, len(ALL_ASPECT_KEYWORDS)))
    
    for keyword in keywords_sample:
        # Skip if keyword is in the adjectives or non-aspect words list
        if keyword.lower() in ADJECTIVES or keyword.lower() in NON_ASPECT_WORDS:
            continue
            
        # Use different templates for variety
        for template in random.sample(templates, min(3, len(templates))):
            example_text = template.format(aspect=keyword)
            
            # Create tokens from the text (use RoBERTa tokenizer)
            # We need to manually annotate the BIO tags
            words = example_text.split()
            tokens = ["<s>"]  # RoBERTa start token
            bio_labels = ["O"] 
            
            # Add word tokens with bio labels
            for word in words:
                word_proc = preprocess_text(word)
                
                if word_proc.lower() == keyword.lower() or keyword.lower() in word_proc.lower():
                    # This word contains our aspect
                    tokens.append(word)
                    bio_labels.append("B-ASP")
                else:
                    tokens.append(word)
                    bio_labels.append("O")
            
            # Add end token
            tokens.append("</s>")  # RoBERTa end token
            bio_labels.append("O")
            
            # Create full entry for dataset
            augmented_data.append({
                "text": example_text,
                "tokens": tokens,
                "bio_labels": bio_labels,
                "aspects": [{"aspect": keyword, "sentiment_id": 1}]  # Default to neutral
            })
            synthetic_count += 1
    
    logger.info(f"Created {synthetic_count} synthetic examples for augmentation")
    return augmented_data

def main():
    args = parse_args()
    
    # Initialize tokenizer
    logger.info(f"Initializing RoBERTa tokenizer from {MODEL_NAME}")
    tokenizer = initialize_tokenizer()
    
    # Set paths
    train_dataset_path = "data/filtered_data/processed_aspect_data_train.json"
    val_dataset_path = "data/filtered_data/processed_aspect_data_val.json"
    
    # Check if we should train the ATE model
    train_ate = not args.train_asc_only
    # Check if we should train the ASC model
    train_asc = not args.train_ate_only
    
    # Train ATE model if needed
    if train_ate:
        # Check if we should resume training
        if args.resume and Path(ASPECT_MODEL_PATH).exists():
            logger.info(f"Resuming ATE training from existing model at {ASPECT_MODEL_PATH}")
            # Use existing model path
            output_dir = ASPECT_MODEL_PATH
        else:
            logger.info("Starting new ATE training")
            output_dir = ASPECT_MODEL_PATH
        
        logger.info("Loading datasets for Aspect Term Extraction...")
        
        # Load the raw data first
        train_data = []
        with open(train_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    train_data.append(json.loads(line))
        
        # Apply preprocessing to filter adjectives if requested
        if not args.include_adjectives:
            logger.info("Filtering adjectives in training data...")
            train_data = preprocess_training_data(train_data, tokenizer, filter_adjectives=True)
            
        # Save the preprocessed data back to a temporary file
        temp_train_path = "data/filtered_data/processed_aspect_data_train_filtered_r.json"
        with open(temp_train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        logger.info(f"Saved preprocessed training data to {temp_train_path}")
        
        # Now load the dataset using the function from model_r.py
        ate_train_dataset = load_aspect_dataset(temp_train_path, tokenizer, ASPECT_LABEL_MAP)
        
        # Apply data augmentation if requested
        if args.augment_data:
            logger.info("Applying data augmentation for ATE training with RoBERTa...")
            augmented_data = augment_aspect_data(tokenizer, filter_adjectives=args.include_adjectives)
            
            # Convert augmented data to dataset
            for item in augmented_data:
                # Process each token for RoBERTa
                processed_text = ' '.join([preprocess_text(token) for token in item['tokens'][1:-1]])  # Skip <s> and </s>
                
                encoded = tokenizer(
                    processed_text,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors=None  # Return Python lists
                )
                
                # Convert BIO labels to IDs
                label_ids = [ASPECT_LABEL_MAP.get(label, 0) for label in item['bio_labels']]
                
                # Pad/truncate label_ids to match input_ids length
                if len(label_ids) < len(encoded['input_ids']):
                    label_ids = label_ids + [0] * (len(encoded['input_ids']) - len(label_ids))
                elif len(label_ids) > len(encoded['input_ids']):
                    label_ids = label_ids[:len(encoded['input_ids'])]
                
                # RoBERTa doesn't use token_type_ids
                dataset_entry = {
                    'input_ids': encoded['input_ids'],
                    'attention_mask': encoded['attention_mask'],
                    'labels': label_ids
                }
                
                ate_train_dataset = ate_train_dataset.add_item(dataset_entry)
            
            logger.info(f"Augmented ATE training dataset size: {len(ate_train_dataset)}")
        
        # Load validation dataset
        ate_val_dataset = load_aspect_dataset(val_dataset_path, tokenizer, ASPECT_LABEL_MAP)
        
        logger.info(f"Starting Aspect Term Extraction training with RoBERTa for {args.epochs} epochs...")
        ate_model, _, ate_metrics = train_aspect_extraction(
            ate_train_dataset, 
            ate_val_dataset, 
            tokenizer,
            num_epochs=args.epochs,
            output_dir=output_dir,
            resume_from_checkpoint=args.resume,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
        
        # Log F1 score as the primary metric
        logger.info(f"ATE training complete with F1 score: {ate_metrics.get('eval_f1', ate_metrics.get('f1', 'N/A'))}")
    else:
        logger.info("Skipping ATE training as requested")
        ate_metrics = {"eval_f1": "N/A"}
    
    # Train ASC model if needed
    if train_asc:
        # Check if we should resume training
        if args.resume and Path(SENTIMENT_MODEL_PATH).exists():
            logger.info(f"Resuming ASC training from existing model at {SENTIMENT_MODEL_PATH}")
            # Use existing model path
            output_dir = SENTIMENT_MODEL_PATH
        else:
            logger.info("Starting new ASC training")
            output_dir = SENTIMENT_MODEL_PATH
        
        logger.info("Loading datasets for Aspect Sentiment Classification...")
        
        # Process the dataset to apply RoBERTa preprocessing
        # For ASC, we need to extract aspects and their sentiments from the train data
        if train_ate:
            # If we just trained ATE, use it to extract aspects for ASC training
            logger.info("Using freshly trained ATE model to extract aspects for ASC training...")
            
            # Load raw data
            raw_data = []
            with open(train_dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        raw_data.append(json.loads(line))
            
            # Extract aspects using the ATE model on each text
            from transformers import pipeline
            ate_pipeline = pipeline("ner", model=ate_model, tokenizer=tokenizer, aggregation_strategy="simple")
            
            # Process each sample to generate better aspect-sentiment pairs
            processed_data = []
            for item in tqdm(raw_data, desc="Extracting aspects for ASC training"):
                text = item.get('text', '')
                if not text:
                    continue
                
                # Preprocess the text for RoBERTa
                text_proc = text  # Keep original for matching
                
                # Extract aspects using the ATE model
                extracted_aspects = ate_pipeline(text_proc)
                
                # Filter out low confidence aspects and non-aspects
                filtered_aspects = [
                    asp for asp in extracted_aspects 
                    if asp['entity_group'] == 'ASP' and asp['score'] >= 0.1
                ]
                
                if not filtered_aspects:
                    # If no aspects were found, skip this example
                    continue
                
                # For each aspect, create an entry with its sentiment
                for aspect_info in item.get('aspects', []):
                    aspect_term = aspect_info.get('aspect', '')
                    if not aspect_term:
                        continue
                    
                    # Clean the aspect for better matching
                    aspect_term_proc = preprocess_text(aspect_term)
                    
                    # Find if this aspect was extracted by the model
                    extracted = False
                    for extracted_asp in filtered_aspects:
                        extracted_word_proc = preprocess_text(extracted_asp['word'])
                        if aspect_term_proc.lower() in extracted_word_proc.lower() or extracted_word_proc.lower() in aspect_term_proc.lower():
                            extracted = True
                            break
                    
                    # Only use aspects that were successfully extracted
                    if extracted:
                        # Get sentiment
                        if 'sentiment_id' in aspect_info:
                            sentiment_id = aspect_info['sentiment_id']
                        elif 'sentiment' in aspect_info:
                            sentiment_text = aspect_info['sentiment'].lower()
                            if sentiment_text == 'negative':
                                sentiment_id = 0
                            elif sentiment_text == 'neutral':
                                sentiment_id = 1
                            elif sentiment_text == 'positive':
                                sentiment_id = 2
                            else:
                                sentiment_id = 1  # Default to neutral
                        else:
                            sentiment_id = 1  # Default to neutral
                        
                        # Add to processed data
                        processed_data.append({
                            'text': text,
                            'aspect': aspect_term,
                            'sentiment_id': sentiment_id
                        })
            
            # Save processed data to a temporary file
            temp_asc_path = "data/filtered_data/processed_asc_data_train_filtered_r.json"
            with open(temp_asc_path, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(json.dumps(item) + '\n')
            
            logger.info(f"Created ASC training data with {len(processed_data)} samples")
            
            # Load ASC datasets
            asc_train_dataset = load_sentiment_dataset(temp_asc_path, tokenizer)
        else:
            # If we didn't train ATE, just use the original data
            asc_train_dataset = load_sentiment_dataset(train_dataset_path, tokenizer)
        
        asc_val_dataset = load_sentiment_dataset(val_dataset_path, tokenizer)
        
        # Log class distribution to understand imbalance
        def get_class_distribution(dataset):
            sentiments = [item['labels'] for item in dataset]
            unique, counts = np.unique(sentiments, return_counts=True)
            distribution = dict(zip(unique, counts))
            return {SENTIMENT_LABELS[i]: count for i, count in distribution.items() if i < len(SENTIMENT_LABELS)}
        
        train_distribution = get_class_distribution(asc_train_dataset)
        logger.info(f"ASC training data sentiment distribution: {train_distribution}")
        
        logger.info(f"Starting Aspect Sentiment Classification training with RoBERTa for {args.epochs} epochs...")
        asc_model, _, asc_metrics = train_aspect_sentiment(
            asc_train_dataset, 
            asc_val_dataset, 
            tokenizer,
            num_epochs=args.epochs,
            output_dir=output_dir,
            resume_from_checkpoint=args.resume,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
        
        # Log Macro F1 as the primary metric for ASC
        logger.info(f"ASC training complete with Macro F1 score: {asc_metrics.get('eval_macro_f1', asc_metrics.get('macro_f1', 'N/A'))}")
    else:
        logger.info("Skipping ASC training as requested")
        asc_metrics = {"eval_macro_f1": "N/A"}
    
    # Print summary of results
    logger.info("Training complete! Summary of results:")
    logger.info(f"ATE Precision: {ate_metrics.get('eval_precision', ate_metrics.get('precision', 'N/A'))}")
    logger.info(f"ATE Recall: {ate_metrics.get('eval_recall', ate_metrics.get('recall', 'N/A'))}")
    logger.info(f"ATE F1 Score: {ate_metrics.get('eval_f1', ate_metrics.get('f1', 'N/A'))}")
    logger.info(f"ASC Macro F1 Score: {asc_metrics.get('eval_macro_f1', asc_metrics.get('macro_f1', 'N/A'))}")
    
    # Report class-wise F1 scores for sentiment analysis
    if train_asc:
        logger.info("ASC Class-wise F1 Scores:")
        logger.info(f"  Negative: {asc_metrics.get('eval_neg_f1', 'N/A')}")
        logger.info(f"  Neutral: {asc_metrics.get('eval_neu_f1', 'N/A')}")
        logger.info(f"  Positive: {asc_metrics.get('eval_pos_f1', 'N/A')}")

if __name__ == "__main__":
    main() 
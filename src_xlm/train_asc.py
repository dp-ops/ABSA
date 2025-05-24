import json
import os
import torch
import logging
import argparse
import numpy as np
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

# Import necessary functions from model.py
from model import (
    # Constants
    MODEL_NAME, SAVED_MODELS_DIR, SENTIMENT_LABELS, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    MAX_LENGTH, PATIENCE, ASPECT_MODEL_PATH, SENTIMENT_MODEL_PATH,
    
    # Functions
    initialize_tokenizer, preprocess_text,
    load_sentiment_dataset, train_aspect_sentiment
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create saved_models directory if it doesn't exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Aspect Sentiment Classification model with XLM-RoBERTa')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of epochs to train the model (default: {NUM_EPOCHS})')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from existing model checkpoint')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help=f'Learning rate for training (default: {LEARNING_RATE})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                        help=f'Patience for early stopping (default: {PATIENCE})')
    parser.add_argument('--gradient_clipping', type=float, default=1.0,
                        help='Gradient clipping value (default: 1.0)')
    parser.add_argument('--data_dir', type=str, default='data/filtered_data_xlm',
                        help='Directory containing the processed data files')
    parser.add_argument('--use_extracted_aspects', action='store_true',
                        help='Use aspects extracted by the ATE model for training')
    
    # Class weight arguments
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights to handle class imbalance')
    parser.add_argument('--class_weights', type=str, default=None,
                        help='Manual class weights as comma-separated values (e.g., "4.0,4.0,1.0" for neg,neu,pos)')
    parser.add_argument('--auto_class_weights', action='store_true',
                        help='Automatically calculate class weights from training data distribution')
    
    return parser.parse_args()

def calculate_class_weights(dataset, method='balanced'):
    """
    Calculate class weights from the dataset to handle class imbalance.
    
    Args:
        dataset: The training dataset
        method: Method for calculating weights ('balanced', 'manual', or None)
        
    Returns:
        torch.Tensor: Class weights tensor
    """
    # Extract labels from dataset
    labels = [item['labels'] for item in dataset]
    labels = np.array(labels)
    
    # Get class distribution
    unique_classes, counts = np.unique(labels, return_counts=True)
    class_distribution = dict(zip(unique_classes, counts))
    
    logger.info(f"Class distribution in training data:")
    for class_id, count in class_distribution.items():
        if class_id < len(SENTIMENT_LABELS):
            logger.info(f"  {SENTIMENT_LABELS[class_id]}: {count} samples")
    
    if method == 'balanced':
        # Use sklearn's compute_class_weight with 'balanced' strategy
        class_weights = compute_class_weight(
            'balanced',
            classes=unique_classes,
            y=labels
        )
        
        # Create tensor with weights for all classes (even if some are missing)
        weights_tensor = torch.ones(len(SENTIMENT_LABELS))
        for i, weight in zip(unique_classes, class_weights):
            if i < len(SENTIMENT_LABELS):
                weights_tensor[i] = weight
        
        logger.info(f"Calculated balanced class weights:")
        for i, weight in enumerate(weights_tensor):
            logger.info(f"  {SENTIMENT_LABELS[i]}: {weight:.4f}")
            
        return weights_tensor
    
    elif method == 'inverse_freq':
        # Use inverse frequency
        total_samples = len(labels)
        weights = []
        
        for class_id in range(len(SENTIMENT_LABELS)):
            if class_id in class_distribution:
                # Weight = total_samples / (num_classes * class_count)
                weight = total_samples / (len(SENTIMENT_LABELS) * class_distribution[class_id])
                weights.append(weight)
            else:
                # If class not present, use default weight of 1.0
                weights.append(1.0)
        
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        
        logger.info(f"Calculated inverse frequency class weights:")
        for i, weight in enumerate(weights_tensor):
            logger.info(f"  {SENTIMENT_LABELS[i]}: {weight:.4f}")
            
        return weights_tensor
    
    else:
        # No weighting
        return None

def parse_manual_class_weights(weights_str):
    """
    Parse manual class weights from command line string.
    
    Args:
        weights_str: Comma-separated string of weights (e.g., "4.0,4.0,1.0")
        
    Returns:
        torch.Tensor: Class weights tensor
    """
    try:
        weights = [float(w.strip()) for w in weights_str.split(',')]
        if len(weights) != len(SENTIMENT_LABELS):
            raise ValueError(f"Expected {len(SENTIMENT_LABELS)} weights, got {len(weights)}")
        
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        
        logger.info(f"Using manual class weights:")
        for i, weight in enumerate(weights_tensor):
            logger.info(f"  {SENTIMENT_LABELS[i]}: {weight:.4f}")
            
        return weights_tensor
    except Exception as e:
        logger.error(f"Error parsing class weights '{weights_str}': {e}")
        logger.error("Expected format: 'weight1,weight2,weight3' (e.g., '4.0,4.0,1.0')")
        return None

def validate_dataset_format(dataset, name="dataset"):
    """
    Validate that the dataset contains valid entries with the correct format.
    
    Args:
        dataset: The dataset to validate
        name: Name of the dataset for logging
        
    Returns:
        True if valid, False otherwise
    """
    logger.info(f"Validating {name} format...")
    
    # Check a small sample of the dataset
    sample_size = min(10, len(dataset))
    
    valid = True
    for i, example in enumerate(dataset.select(range(sample_size))):
        # Check required fields
        required_fields = ["input_ids", "attention_mask", "labels"]
        for field in required_fields:
            if field not in example:
                logger.error(f"Example {i} in {name} missing required field '{field}'")
                valid = False
                continue
                
        # Check that fields are lists of integers or integers
        for field in required_fields:
            if field == "labels":
                if not isinstance(example[field], int):
                    logger.error(f"Field '{field}' in example {i} of {name} is not an integer but {type(example[field])}")
                    valid = False
            else:
                if not isinstance(example[field], list):
                    logger.error(f"Field '{field}' in example {i} of {name} is not a list but {type(example[field])}")
                    valid = False
                elif not all(isinstance(x, int) for x in example[field]):
                    logger.error(f"Field '{field}' in example {i} of {name} contains non-integer values")
                    valid = False
    
    if valid:
        logger.info(f"{name} format validation passed")
    else:
        logger.error(f"{name} format validation failed")
        
    return valid

def extract_aspects_for_asc(raw_data, ate_model, tokenizer):
    """
    Use the trained ATE model to extract aspects for ASC training.
    This helps create a more realistic dataset by using the actual aspects
    that would be extracted during inference.
    
    Args:
        raw_data: List of raw data items
        ate_model: Trained ATE model
        tokenizer: Tokenizer to use
        
    Returns:
        List of processed data items with extracted aspects
    """
    from transformers import pipeline
    logger.info("Extracting aspects using the ATE model for ASC training...")
    
    # Create NER pipeline for aspect extraction
    ner_pipeline = pipeline(
        "ner", 
        model=ate_model, 
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    
    processed_data = []
    
    for item in tqdm(raw_data, desc="Extracting aspects"):
        text = item.get('text', '')
        if not text:
            continue
        
        # Extract aspects using the ATE model
        extracted_aspects = ner_pipeline(text)
        
        # Filter to get only ASP entities with reasonable confidence
        filtered_aspects = [
            asp for asp in extracted_aspects 
            if asp['entity_group'] == 'ASP' and asp['score'] >= 0.1
        ]
        
        # For each extracted aspect, create an entry
        for aspect_info in filtered_aspects:
            aspect_term = aspect_info['word']
            
            # Try to find the sentiment for this aspect in the gold data
            sentiment_id = 1  # Default to neutral
            
            for gold_aspect in item.get('aspects', []):
                if gold_aspect.get('aspect', '').lower() == aspect_term.lower():
                    # Found a matching aspect, get its sentiment
                    if 'sentiment_id' in gold_aspect:
                        sentiment_id = gold_aspect['sentiment_id']
                    elif 'sentiment' in gold_aspect:
                        sentiment_text = gold_aspect['sentiment'].lower()
                        if sentiment_text == 'negative':
                            sentiment_id = 0
                        elif sentiment_text == 'neutral':
                            sentiment_id = 1
                        elif sentiment_text == 'positive':
                            sentiment_id = 2
                    break
            
            # Add to processed data
            processed_data.append({
                'text': text,
                'aspect': aspect_term,
                'sentiment_id': sentiment_id
            })
        
        # Also include all gold aspects for better coverage
        for gold_aspect in item.get('aspects', []):
            aspect_term = gold_aspect.get('aspect', '')
            if not aspect_term:
                continue
                
            # Get sentiment
            if 'sentiment_id' in gold_aspect:
                sentiment_id = gold_aspect['sentiment_id']
            elif 'sentiment' in gold_aspect:
                sentiment_text = gold_aspect['sentiment'].lower()
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
    
    # Remove duplicates
    unique_data = {}
    for item in processed_data:
        key = f"{item['text']}|{item['aspect']}"
        unique_data[key] = item
    
    processed_data = list(unique_data.values())
    
    logger.info(f"Created ASC training data with {len(processed_data)} samples")
    return processed_data

def main():
    args = parse_args()
    
    # Initialize tokenizer
    logger.info(f"Initializing XLM-RoBERTa tokenizer from {MODEL_NAME}")
    tokenizer = initialize_tokenizer()
    
    # Set paths
    train_dataset_path = f"{args.data_dir}/processed_aspect_data_train.json"
    val_dataset_path = f"{args.data_dir}/processed_aspect_data_val.json"
    
    # Check if the data directory exists
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory '{args.data_dir}' not found. Please check the path.")
        available_dirs = [d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d)) and d.startswith('filter')]
        if available_dirs:
            logger.info(f"Available data directories: {available_dirs}")
        return
    
    # Check if the train and validation files exist
    if not os.path.exists(train_dataset_path):
        logger.error(f"Training dataset not found at '{train_dataset_path}'. Please check the file path.")
        return
    
    if not os.path.exists(val_dataset_path):
        logger.error(f"Validation dataset not found at '{val_dataset_path}'. Please check the file path.")
        return
    
    # Load raw data for processing
    train_data = []
    with open(train_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                train_data.append(json.loads(line))
    
    # Check if we should use the ATE model to extract aspects
    if args.use_extracted_aspects:
        try:
            from transformers import AutoModelForTokenClassification
            logger.info(f"Loading ATE model from {ASPECT_MODEL_PATH} for aspect extraction")
            ate_model = AutoModelForTokenClassification.from_pretrained(ASPECT_MODEL_PATH)
            
            # Extract aspects using the ATE model
            processed_train_data = extract_aspects_for_asc(train_data, ate_model, tokenizer)
            
            # Save processed data to a temporary file
            temp_asc_path = f"{args.data_dir}/processed_asc_data_train.json"
            with open(temp_asc_path, 'w', encoding='utf-8') as f:
                for item in processed_train_data:
                    f.write(json.dumps(item) + '\n')
            
            # Load the processed data
            asc_train_dataset = load_sentiment_dataset(temp_asc_path, tokenizer)
        except Exception as e:
            logger.warning(f"Error loading ATE model: {e}. Using original data for ASC training.")
            asc_train_dataset = load_sentiment_dataset(train_dataset_path, tokenizer)
    else:
        # Use the original data directly
        asc_train_dataset = load_sentiment_dataset(train_dataset_path, tokenizer)
    
    # Load validation dataset
    asc_val_dataset = load_sentiment_dataset(val_dataset_path, tokenizer)
    
    # Validate dataset formats
    train_valid = validate_dataset_format(asc_train_dataset, "ASC training dataset")
    val_valid = validate_dataset_format(asc_val_dataset, "ASC validation dataset")
    
    if not train_valid or not val_valid:
        logger.error("Dataset validation failed. Exiting.")
        return
    
    # Calculate class weights if requested
    class_weights = None
    if args.use_class_weights or args.auto_class_weights or args.class_weights:
        logger.info("Setting up class weights for handling sentiment class imbalance...")
        
        if args.class_weights:
            # Use manual weights
            class_weights = parse_manual_class_weights(args.class_weights)
            if class_weights is None:
                logger.error("Failed to parse manual class weights. Exiting.")
                return
        elif args.auto_class_weights:
            # Calculate weights automatically using balanced strategy
            class_weights = calculate_class_weights(asc_train_dataset, method='balanced')
        else:
            # Default to balanced weights if --use_class_weights is specified
            class_weights = calculate_class_weights(asc_train_dataset, method='balanced')
        
        if class_weights is not None:
            logger.info("Class weights will be applied during training to handle imbalance.")
        else:
            logger.warning("Class weights could not be calculated. Training without weights.")
    else:
        logger.info("No class weights specified. Training with standard cross-entropy loss.")
    
    # Train model
    logger.info(f"Starting ASC training for {args.epochs} epochs...")
    model, tokenizer, eval_metrics = train_aspect_sentiment(
        asc_train_dataset, 
        asc_val_dataset, 
        tokenizer,
        num_epochs=args.epochs,
        output_dir=SENTIMENT_MODEL_PATH,
        resume_from_checkpoint_cli_flag=args.resume,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_clipping=args.gradient_clipping,
        class_weights=class_weights,
        patience=args.patience,
        lr_reduction_patience_asc=args.patience  # Use the exact --patience value
    )
    
    # Log final metrics
    logger.info(f"Training complete! Final Macro F1 score: {eval_metrics.get('eval_macro_f1', 'N/A')}")

if __name__ == "__main__":
    main()

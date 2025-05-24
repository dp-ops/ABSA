import json
import os
import torch
import logging
import argparse
from pathlib import Path
from datasets import Dataset
from tqdm import tqdm

# Import necessary functions from model.py instead of model_xlm.py
from model import (
    # Constants
    MODEL_NAME, SAVED_MODELS_DIR, ASPECT_LABEL_MAP, ASPECT_LABEL_MAP_INVERSE,
    MAX_LENGTH, PATIENCE, ASPECT_MODEL_PATH, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE,
    
    # Functions
    initialize_tokenizer, preprocess_text, 
    load_aspect_dataset, enhanced_align_tokens_and_labels, convert_aligned_labels_to_ids,
    train_aspect_extraction, back_translate
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create saved_models_xlm directory if it doesn't exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train Aspect Term Extraction model with XLM-RoBERTa and CRF')
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
    parser.add_argument('--use_crf', action='store_true', default=True,
                        help='Use CRF layer on top of XLM-RoBERTa (default: True)')
    parser.add_argument('--augment_data', action='store_true',
                        help='Use data augmentation with back-translation')
    parser.add_argument('--data_dir', type=str, default='data/filtered_data_xlm',
                        help='Directory containing the processed data files')
    return parser.parse_args()

def augment_with_back_translation(train_data, tokenizer, label_map=ASPECT_LABEL_MAP):
    """
    Augment training data with back-translation to increase variety.
    This creates new samples by translating the text to another language and back.
    """
    logger.info("Augmenting training data with back-translation...")
    
    augmented_data = []
    
    # Only augment a subset of the data to keep training time reasonable
    max_samples = min(10, len(train_data))  # Reduced for testing
    
    for item in tqdm(train_data[:max_samples], desc="Back-translating"):
        # Skip items without text
        if 'text' not in item:
            continue
            
        # Get original text
        orig_text = item['text']
        
        # Apply back-translation
        translated_text = back_translate(orig_text, source_lang="el", target_lang="en")
        
        # Skip if translation failed or returned the same text
        if not translated_text or translated_text == orig_text:
            continue
            
        # If the item has tokens and BIO labels, align them with the translated text
        if 'tokens' in item and 'bio_labels' in item:
            # Create a new item with translated text
            new_item = {
                'text': translated_text,
                'aspects': item['aspects'] if 'aspects' in item else []  # Keep the same aspects
            }
            
            # Tokenize and align with the new text
            encoding = tokenizer(
                translated_text,
                padding="max_length",
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors=None  # Return Python lists
            )
            
            # Initialize all labels as O (0)
            labels = [0] * len(encoding['input_ids'])
            
            # Ensure input_ids and attention_mask are lists of integers
            if not isinstance(encoding['input_ids'], list) or not all(isinstance(x, int) for x in encoding['input_ids']):
                logger.warning(f"Input IDs not in correct format. Got: {type(encoding['input_ids'])}")
                continue
                
            if not isinstance(encoding['attention_mask'], list) or not all(isinstance(x, int) for x in encoding['attention_mask']):
                logger.warning(f"Attention mask not in correct format. Got: {type(encoding['attention_mask'])}")
                continue
                
            if not isinstance(labels, list) or not all(isinstance(x, int) for x in labels):
                logger.warning(f"Labels not in correct format. Got: {type(labels)}")
                continue
            
            # Create entry
            entry = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': labels
            }
            augmented_data.append(entry)
    
    logger.info(f"Created {len(augmented_data)} augmented samples with back-translation")
    return augmented_data

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
                
        # Check that fields are lists of integers
        for field in required_fields:
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
    
    # Load datasets
    logger.info("Loading datasets for Aspect Term Extraction...")
    train_dataset = load_aspect_dataset(train_dataset_path, tokenizer)
    val_dataset = load_aspect_dataset(val_dataset_path, tokenizer)
    
    # Validate dataset formats
    train_valid = validate_dataset_format(train_dataset, "training dataset")
    val_valid = validate_dataset_format(val_dataset, "validation dataset")
    
    if not train_valid or not val_valid:
        logger.error("Dataset validation failed. Exiting.")
        return
    
    # Apply data augmentation if requested
    if args.augment_data:
        logger.info("Applying data augmentation with back-translation...")
        
        # Load raw data for augmentation
        train_data = []
        with open(train_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    train_data.append(json.loads(line))
        
        # Augment data with back-translation
        augmented_data = augment_with_back_translation(train_data, tokenizer)
        
        # Convert to Dataset format
        if augmented_data:
            augmented_dataset = Dataset.from_list(augmented_data)
            
            # Combine with original dataset
            train_dataset = train_dataset.filter(lambda x: True)  # Create a copy
            
            # For debugging
            logger.info(f"Original dataset size: {len(train_dataset)}")
            logger.info(f"Augmented dataset size: {len(augmented_dataset)}")
            
            # Validate augmented dataset
            aug_valid = validate_dataset_format(augmented_dataset, "augmented dataset")
            if not aug_valid:
                logger.error("Augmented dataset validation failed. Using only original dataset.")
            else:
                # Merge datasets
                train_dataset = train_dataset.map(lambda x: x)  # Make a copy
                augmented_dataset = augmented_dataset.map(lambda x: x)  # Make a copy
                
                # Concatenate datasets
                from datasets import concatenate_datasets
                train_dataset = concatenate_datasets([train_dataset, augmented_dataset])
                
                logger.info(f"Combined dataset size: {len(train_dataset)}")
    
    # Train model
    logger.info(f"Starting training for {args.epochs} epochs...")
    model, tokenizer, eval_metrics = train_aspect_extraction(
        train_dataset, 
        val_dataset, 
        tokenizer,
        num_epochs=args.epochs,
        resume_from_checkpoint_cli_flag=args.resume,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        use_crf=args.use_crf,
        patience=args.patience,
        gradient_clipping=args.gradient_clipping
    )
    
    # Log final metrics
    logger.info(f"Training complete! Final F1 score: {eval_metrics.get('eval_f1', 'N/A')}")

if __name__ == "__main__":
    main() 
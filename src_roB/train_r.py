import json
import os
import numpy as np
import logging
import argparse
from pathlib import Path
from datasets import Dataset, concatenate_datasets
import random
from tqdm import tqdm

# Import necessary functions from model_r.py
from model_r import (
    # Constants
    MODEL_NAME, SAVED_MODELS_DIR, ASPECT_LABEL_MAP, ASPECT_LABEL_MAP_INVERSE,
    SENTIMENT_LABELS, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, MAX_LENGTH, PATIENCE,
    ASPECT_MODEL_PATH, SENTIMENT_MODEL_PATH, ASPECT_KEYWORDS_MAP, ALL_ASPECT_KEYWORDS,
    STOPWORDS, ADJECTIVES,
    
    # Functions
    initialize_tokenizer, preprocess_text, enhanced_preprocess_text,
    load_aspect_dataset, load_sentiment_dataset,
    train_aspect_extraction, train_aspect_sentiment,
    enhanced_align_tokens_and_labels, convert_aligned_labels_to_ids
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

# Hardcoded class weights based on dataset statistics
# Default values from logs: [0.36172401, 4.84184338, 34.56806109]
# weights: 0 b-asp, 1 i-asp, 2 o
# DEFAULT_CLASS_WEIGHTS = [0.36, 4.84, 34.57]
DEFAULT_CLASS_WEIGHTS = [0.3, 8, 3]  # O, B-ASP, I-ASP

def parse_args():
    parser = argparse.ArgumentParser(description='Train ABSA models with RoBERTa')
    parser.add_argument('--train_ate_epochs', type=int, default=None,
                        help=f'Number of epochs to train the Aspect Term Extraction model. If not provided, ATE training will be skipped.')
    parser.add_argument('--train_asc_epochs', type=int, default=None,
                        help=f'Number of epochs to train the Aspect Sentiment Classification model. If not provided, ASC training will be skipped.')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from existing model checkpoints')
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
    parser.add_argument('--data_dir', type=str, default='data/filtered_data_r',
                        help='Directory containing the processed data files (default: data/filtered_data_r)')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use focal loss instead of cross-entropy')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                        help='Number of steps to accumulate gradients (default: 1)')
    parser.add_argument('--class_weights', type=str, default=None,
                        help=f'Comma-separated class weights for O, B-ASP, I-ASP (default: {",".join(map(str, DEFAULT_CLASS_WEIGHTS))})')
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
                    # Use enhanced preprocessing for better handling of Greek text
                    token_proc = enhanced_preprocess_text(token).lower()
                    
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
    train_dataset_path = f"{args.data_dir}/processed_aspect_data_train.json"
    val_dataset_path = f"{args.data_dir}/processed_aspect_data_val.json"
    
    # Parse class weights if provided
    class_weights = DEFAULT_CLASS_WEIGHTS
    if args.class_weights:
        try:
            class_weights = [float(w) for w in args.class_weights.split(',')]
            if len(class_weights) != 3:
                logger.warning(f"Expected 3 class weights, got {len(class_weights)}. Using default weights.")
                class_weights = DEFAULT_CLASS_WEIGHTS
        except ValueError:
            logger.warning(f"Invalid class weights format. Using default weights.")
    
    # Convert class weights to tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    logger.info(f"Using class weights: {class_weights}")
    
    # Train ATE model if epochs is provided
    ate_model = None
    if args.train_ate_epochs is not None:
        # Check if we should resume training
        if args.resume and Path(ASPECT_MODEL_PATH).exists():
            logger.info(f"Resuming ATE training from existing model at {ASPECT_MODEL_PATH}")
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
        temp_train_path = f"{args.data_dir}/processed_aspect_data_train_filtered_r.json"
        with open(temp_train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item) + '\n')
        
        # Create dataset using enhanced alignment for better handling of Greek text
        formatted_data = []
        for item in train_data:
            if 'tokens' in item and 'bio_labels' in item:
                # Use enhanced alignment
                aligned_tokens, aligned_labels = enhanced_align_tokens_and_labels(
                    item['tokens'], item['bio_labels'], tokenizer
                )
                
                # Convert aligned labels to IDs
                label_ids = convert_aligned_labels_to_ids(aligned_labels, ASPECT_LABEL_MAP)
                
                # Tokenize directly for encoding
                encoding = tokenizer(
                    item.get('text', ' '.join(item['tokens'])),
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
                
                # Create entry
                entry = {
                    'input_ids': encoding['input_ids'],
                    'attention_mask': encoding['attention_mask'],
                    'labels': label_ids
                }
                formatted_data.append(entry)
                
        ate_train_dataset = Dataset.from_list(formatted_data)
        
        # Apply data augmentation if requested
        if args.augment_data:
            logger.info("Applying data augmentation for ATE training...")
            augmented_data = augment_aspect_data(tokenizer, filter_adjectives=not args.include_adjectives)
            
            # Convert augmented data to dataset
            formatted_augmented_data = []
            for item in augmented_data:
                # Use enhanced alignment for synthetic examples too
                aligned_tokens, aligned_labels = enhanced_align_tokens_and_labels(
                    item['tokens'], item['bio_labels'], tokenizer
                )
                
                # Convert aligned labels to IDs
                label_ids = convert_aligned_labels_to_ids(aligned_labels, ASPECT_LABEL_MAP)
                
                # Tokenize directly for encoding
                encoding = tokenizer(
                    item.get('text', ' '.join(item['tokens'])),
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
                
                # Create entry
                entry = {
                    'input_ids': encoding['input_ids'],
                    'attention_mask': encoding['attention_mask'],
                    'labels': label_ids
                }
                formatted_augmented_data.append(entry)
                
            # Add augmented data to training dataset
            augmented_dataset = Dataset.from_list(formatted_augmented_data)
            ate_train_dataset = concatenate_datasets([ate_train_dataset, augmented_dataset])
            
            logger.info(f"Augmented ATE training dataset size: {len(ate_train_dataset)}")
        
        # Load validation dataset with same preprocessing options
        # Load raw validation data
        val_data = []
        with open(val_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    val_data.append(json.loads(line))
        
        # Apply preprocessing
        val_data = preprocess_training_data(val_data, tokenizer, filter_adjectives=not args.include_adjectives)
        
        # Format using enhanced alignment
        val_formatted_data = []
        for item in val_data:
            if 'tokens' in item and 'bio_labels' in item:
                # Use enhanced alignment
                aligned_tokens, aligned_labels = enhanced_align_tokens_and_labels(
                    item['tokens'], item['bio_labels'], tokenizer
                )
                
                # Convert aligned labels to IDs
                label_ids = convert_aligned_labels_to_ids(aligned_labels, ASPECT_LABEL_MAP)
                
                # Tokenize directly for encoding
                encoding = tokenizer(
                    item.get('text', ' '.join(item['tokens'])),
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
                
                # Create entry
                entry = {
                    'input_ids': encoding['input_ids'],
                    'attention_mask': encoding['attention_mask'],
                    'labels': label_ids
                }
                val_formatted_data.append(entry)
                
        ate_val_dataset = Dataset.from_list(val_formatted_data)
        
        # Create dictionary of training options
        training_options = {
            "use_focal_loss": args.use_focal_loss,
            "class_weights": class_weights_tensor,
            "gradient_accumulation_steps": args.gradient_accumulation
        }
        
        logger.info(f"Starting Aspect Term Extraction training for {args.train_ate_epochs} epochs...")
        ate_model, _, ate_metrics = train_aspect_extraction(
            ate_train_dataset, 
            ate_val_dataset, 
            tokenizer,
            num_epochs=args.train_ate_epochs,
            output_dir=output_dir,
            resume_from_checkpoint=args.resume,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            **training_options
        )
        
        # Log F1 score as the primary metric
        logger.info(f"ATE training complete with F1 score: {ate_metrics.get('eval_f1', ate_metrics.get('f1', 'N/A'))}")
        
        # Explicitly load the best checkpoint based on validation F1 score
        best_checkpoint_dir = None
        
        # Check if there are checkpoints saved
        checkpoints_dir = Path(f"{output_dir}/checkpoints")
        if checkpoints_dir.exists():
            logger.info("Looking for best checkpoint based on validation F1 score...")
            
            # Find all checkpoint directories
            checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
            
            if checkpoint_dirs:
                best_f1 = -float('inf')
                
                # Iterate through checkpoints to find the best one
                for checkpoint_dir in checkpoint_dirs:
                    # Check if there's a trainer_state.json file
                    trainer_state_file = checkpoint_dir / "trainer_state.json"
                    if trainer_state_file.exists():
                        try:
                            with open(trainer_state_file, 'r') as f:
                                trainer_state = json.load(f)
                                
                            # Extract the best metric value
                            if 'best_metric' in trainer_state:
                                checkpoint_f1 = trainer_state['best_metric']
                                
                                if checkpoint_f1 > best_f1:
                                    best_f1 = checkpoint_f1
                                    best_checkpoint_dir = checkpoint_dir
                        except Exception as e:
                            logger.warning(f"Error reading checkpoint state: {e}")
                
                if best_checkpoint_dir:
                    logger.info(f"Found best checkpoint with F1 score {best_f1:.4f} at {best_checkpoint_dir}")
                    
                    # Load the best checkpoint for ATE model
                    try:
                        logger.info(f"Loading best checkpoint from {best_checkpoint_dir}")
                        ate_model = AutoModelForTokenClassification.from_pretrained(str(best_checkpoint_dir))
                        
                        # Save this as the final model
                        ate_model.save_pretrained(output_dir)
                        logger.info(f"Saved best checkpoint as the final model to {output_dir}")
                    except Exception as e:
                        logger.error(f"Error loading best checkpoint: {e}")
                else:
                    logger.warning("Could not find a checkpoint with F1 score. Using the final model.")
            else:
                logger.warning("No checkpoints found. Using the final model.")
    else:
        logger.info("Skipping ATE training as no epochs were specified")
        ate_metrics = {"eval_f1": "N/A"}
    
    # Train ASC model if epochs is provided
    if args.train_asc_epochs is not None:
        # Check if we should resume training
        if args.resume and Path(SENTIMENT_MODEL_PATH).exists():
            logger.info(f"Resuming ASC training from existing model at {SENTIMENT_MODEL_PATH}")
            output_dir = SENTIMENT_MODEL_PATH
        else:
            logger.info("Starting new ASC training")
            output_dir = SENTIMENT_MODEL_PATH
        
        logger.info("Loading datasets for Aspect Sentiment Classification...")
        
        # Process the dataset to apply preprocessing
        # For ASC, we need to extract aspects and their sentiments from the train data
        if ate_model is not None:
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
                
                # Extract aspects using the ATE model
                extracted_aspects = ate_pipeline(text)
                
                # Filter out low confidence aspects and non-aspects
                filtered_aspects = [
                    asp for asp in extracted_aspects 
                    if asp['entity_group'] == 'ASP' and asp['score'] >= 0.1
                ]
                
                # For each aspect in the gold standard data, create an entry with its sentiment
                for aspect_info in item.get('aspects', []):
                    aspect_term = aspect_info.get('aspect', '')
                    if not aspect_term:
                        continue
                    
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
                    
                    # Add to processed data (using all gold standard aspects regardless of extraction)
                    processed_data.append({
                        'text': text,
                        'aspect': aspect_term,
                        'sentiment_id': sentiment_id
                    })
            
            # Save processed data to a temporary file
            temp_asc_path = f"{args.data_dir}/processed_asc_data_train_filtered_r.json"
            with open(temp_asc_path, 'w', encoding='utf-8') as f:
                for item in processed_data:
                    f.write(json.dumps(item) + '\n')
            
            logger.info(f"Created ASC training data with {len(processed_data)} samples")
            
            # Load ASC datasets
            asc_train_dataset = load_sentiment_dataset(temp_asc_path, tokenizer)
        else:
            # If we didn't train ATE, try to load the existing model for aspect extraction
            try:
                from transformers import AutoModelForTokenClassification
                logger.info(f"Loading existing ATE model from {ASPECT_MODEL_PATH} for aspect extraction")
                ate_model = AutoModelForTokenClassification.from_pretrained(ASPECT_MODEL_PATH)
                
                # Extract aspects using the loaded ATE model
                from transformers import pipeline
                ate_pipeline = pipeline("ner", model=ate_model, tokenizer=tokenizer, aggregation_strategy="simple")
                
                # Load raw data
                raw_data = []
                with open(train_dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():  # Skip empty lines
                            raw_data.append(json.loads(line))
                
                # Process each sample to generate better aspect-sentiment pairs
                processed_data = []
                for item in tqdm(raw_data, desc="Extracting aspects for ASC training"):
                    text = item.get('text', '')
                    if not text:
                        continue
                    
                    # For each aspect in the gold standard data, create an entry with its sentiment
                    for aspect_info in item.get('aspects', []):
                        aspect_term = aspect_info.get('aspect', '')
                        if not aspect_term:
                            continue
                        
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
                        
                        # Add to processed data (using all gold standard aspects regardless of extraction)
                        processed_data.append({
                            'text': text,
                            'aspect': aspect_term,
                            'sentiment_id': sentiment_id
                        })
                
                # Save processed data to a temporary file
                temp_asc_path = f"{args.data_dir}/processed_asc_data_train_filtered_r.json"
                with open(temp_asc_path, 'w', encoding='utf-8') as f:
                    for item in processed_data:
                        f.write(json.dumps(item) + '\n')
                
                logger.info(f"Created ASC training data with {len(processed_data)} samples")
                
                # Load ASC datasets
                asc_train_dataset = load_sentiment_dataset(temp_asc_path, tokenizer)
            except Exception as e:
                logger.warning(f"Could not load existing ATE model: {e}. Using original data for ASC training.")
                # If we couldn't load the ATE model, just use the original data
                asc_train_dataset = load_sentiment_dataset(train_dataset_path, tokenizer)
        
        asc_val_dataset = load_sentiment_dataset(val_dataset_path, tokenizer)
        
        logger.info(f"Starting Aspect Sentiment Classification training for {args.train_asc_epochs} epochs...")
        asc_model, _, asc_metrics = train_aspect_sentiment(
            asc_train_dataset, 
            asc_val_dataset, 
            tokenizer,
            num_epochs=args.train_asc_epochs,
            output_dir=output_dir,
            resume_from_checkpoint=args.resume,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
        
        # Log Macro F1 as the primary metric for ASC
        logger.info(f"ASC training complete with Macro F1 score: {asc_metrics.get('eval_macro_f1', asc_metrics.get('macro_f1', 'N/A'))}")
        
        # Explicitly load the best checkpoint based on validation macro F1 score
        best_checkpoint_dir = None
        
        # Check if there are checkpoints saved
        checkpoints_dir = Path(f"{output_dir}/checkpoints")
        if checkpoints_dir.exists():
            logger.info("Looking for best checkpoint based on validation Macro F1 score...")
            
            # Find all checkpoint directories
            checkpoint_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
            
            if checkpoint_dirs:
                best_f1 = -float('inf')
                
                # Iterate through checkpoints to find the best one
                for checkpoint_dir in checkpoint_dirs:
                    # Check if there's a trainer_state.json file
                    trainer_state_file = checkpoint_dir / "trainer_state.json"
                    if trainer_state_file.exists():
                        try:
                            with open(trainer_state_file, 'r') as f:
                                trainer_state = json.load(f)
                                
                            # Extract the best metric value
                            if 'best_metric' in trainer_state:
                                checkpoint_f1 = trainer_state['best_metric']
                                
                                if checkpoint_f1 > best_f1:
                                    best_f1 = checkpoint_f1
                                    best_checkpoint_dir = checkpoint_dir
                        except Exception as e:
                            logger.warning(f"Error reading checkpoint state: {e}")
                
                if best_checkpoint_dir:
                    logger.info(f"Found best checkpoint with Macro F1 score {best_f1:.4f} at {best_checkpoint_dir}")
                    
                    # Load the best checkpoint for ASC model
                    try:
                        logger.info(f"Loading best checkpoint from {best_checkpoint_dir}")
                        asc_model = AutoModelForSequenceClassification.from_pretrained(str(best_checkpoint_dir))
                        
                        # Save this as the final model
                        asc_model.save_pretrained(output_dir)
                        logger.info(f"Saved best checkpoint as the final model to {output_dir}")
                    except Exception as e:
                        logger.error(f"Error loading best checkpoint: {e}")
                else:
                    logger.warning("Could not find a checkpoint with Macro F1 score. Using the final model.")
            else:
                logger.warning("No checkpoints found. Using the final model.")
    else:
        logger.info("Skipping ASC training as no epochs were specified")
        asc_metrics = {"eval_macro_f1": "N/A"}
    
    # Print summary of results
    logger.info("Training complete! Summary of results:")
    if args.train_ate_epochs is not None:
        logger.info(f"ATE F1 Score: {ate_metrics.get('eval_f1', ate_metrics.get('f1', 'N/A'))}")
    if args.train_asc_epochs is not None:
        logger.info(f"ASC Macro F1 Score: {asc_metrics.get('eval_macro_f1', asc_metrics.get('macro_f1', 'N/A'))}")

if __name__ == "__main__":
    # Add missing import
    import torch
    main() 
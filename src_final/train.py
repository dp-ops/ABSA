import os
import logging
import argparse
import shutil

from model import (
    # Constants
    MODEL_NAME, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, 
    ATC_MODEL_PATH, ASC_MODEL_PATH,
    # Functions
    initialize_tokenizer,
    load_atc_dataset, train_aspect_classification,
    load_asc_dataset, train_aspect_sentiment
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fixed class weights for ASC: [negative, neutral, positive]
# Calculated from actual data distribution (10.47% neg, 16.41% neu, 73.13% pos)
# Using inverse frequency weighting: total_samples / (num_classes * class_count)
# Negative: 20444/(3*2140) ≈ 3.18, Neutral: 20444/(3*3354) ≈ 2.03, Positive: 20444/(3*14950) ≈ 0.46
ASC_CLASS_WEIGHTS_VALUES = [2.0, 2.0, 1]  #[6.0, 4.0, 0.5]

def parse_args():
    parser = argparse.ArgumentParser(description='Train Aspect Term Classification (ATC) or Aspect Sentiment Classification (ASC) model.')
    
    # Model selection
    parser.add_argument('--train_atc', '--atc', action='store_true', help='Train the Aspect Term Classification (ATC) model.')
    parser.add_argument('--train_asc', '--asc', action='store_true', help='Train the Aspect Sentiment Classification (ASC) model.')
    
    # Training parameters
    parser.add_argument('--epochs', '-e', type=int, default=NUM_EPOCHS,
                        help=f'Number of epochs to train (default: {NUM_EPOCHS})')
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5, optimized for imbalanced data)')
    parser.add_argument('--batch_size', '--b', type=int, default=8,
                        help='Batch size (default: 8, smaller for imbalanced data)')
    parser.add_argument('--gradient_clipping', '--gc', type=float, default=0.5,
                        help='Gradient clipping norm (default: 0.5, tighter for stability)')
    parser.add_argument('--resume', action='store_true', 
                        help='Resume training from the latest checkpoint in the model output directory.')
    parser.add_argument('--data_dir', '-d', type=str, default='data/f_data_bert_lemma',
                        help='Directory containing the processed data files (e.g., data/f_data_bert or data/f_data_bert_lemma).')
    parser.add_argument('--asc_use_class_weights', action='store_true',
                        help=f'Use predefined class weights {ASC_CLASS_WEIGHTS_VALUES} for ASC training to handle imbalance.')
    parser.add_argument('--augmented_data_asc', action='store_true',
                        help='Use augmented datasets for ASC training. Will append "_augmented" to the data_dir to load augmented data.')
    
    return parser.parse_args()

def main():
    args = parse_args()

    if not args.train_atc and not args.train_asc:
        logger.error("Please specify which model to train: --train_atc or --train_asc")
        return
    
    if args.train_atc and args.train_asc:
        logger.error("Please choose only one model to train at a time: --train_atc OR --train_asc")
        return

    # Initialize tokenizer
    logger.info(f"Initializing tokenizer from {MODEL_NAME}")
    tokenizer = initialize_tokenizer()

    # Determine data directory based on training type and augmentation
    data_dir = args.data_dir
    
    # For ASC training, check if augmented data should be used
    if args.train_asc and args.augmented_data_asc:
        data_dir = args.data_dir + "_augmented"
        logger.info(f"Using augmented data directory for ASC training: {data_dir}")
        
        # Check if augmented directory exists
        if not os.path.exists(data_dir):
            logger.error(f"Augmented data directory not found: {data_dir}")
            logger.error("Please run data_prep.py with --create_augmented flag first to create augmented datasets.")
            return
    elif args.train_asc and not args.augmented_data_asc:
        logger.info(f"Using standard (non-augmented) data directory for ASC training: {data_dir}")
    else:
        # ATC training always uses standard data
        logger.info(f"Using standard data directory for ATC training: {data_dir}")

    # Construct paths to data files
    train_file = os.path.join(data_dir, "train_data.json")
    val_file = os.path.join(data_dir, "val_data.json")

    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return
    if not os.path.exists(train_file):
        logger.error(f"Train data file not found: {train_file}")
        return
    if not os.path.exists(val_file):
        logger.error(f"Validation data file not found: {val_file}")
        return

    # Delete existing models if not resuming
    if not args.resume:
        if args.train_atc and os.path.exists(ATC_MODEL_PATH):
            logger.info(f"Deleting existing ATC model directory: {ATC_MODEL_PATH}")
            shutil.rmtree(ATC_MODEL_PATH)
        elif args.train_asc and os.path.exists(ASC_MODEL_PATH):
            logger.info(f"Deleting existing ASC model directory: {ASC_MODEL_PATH}")
            shutil.rmtree(ASC_MODEL_PATH)

    # Add check for output directories
    os.makedirs(ATC_MODEL_PATH, exist_ok=True)
    os.makedirs(ASC_MODEL_PATH, exist_ok=True)

    if args.train_atc:
        logger.info("--- Training Aspect Term Classification (ATC) Model ---")
        logger.info(f"Loading ATC dataset from {data_dir}")
        train_dataset = load_atc_dataset(train_file, tokenizer)
        val_dataset = load_atc_dataset(val_file, tokenizer)

        logger.info(f"Starting ATC training for {args.epochs} epochs...")
        _, _, eval_metrics = train_aspect_classification(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            num_epochs=args.epochs,
            output_dir=ATC_MODEL_PATH,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_clipping_norm=args.gradient_clipping,
            resume_from_checkpoint=args.resume
        )
        logger.info(f"ATC Training complete. Evaluation F1: {eval_metrics.get('eval_f1', 'N/A')}")

    elif args.train_asc:
        augmentation_status = "with augmented data" if args.augmented_data_asc else "with standard data"
        logger.info(f"--- Training Aspect Sentiment Classification (ASC) Model {augmentation_status} ---")
        logger.info(f"Loading ASC dataset from {data_dir}")
        train_dataset = load_asc_dataset(train_file, tokenizer)
        val_dataset = load_asc_dataset(val_file, tokenizer)

        asc_weights_to_use = None
        if args.asc_use_class_weights:
            asc_weights_to_use = ASC_CLASS_WEIGHTS_VALUES
            logger.info(f"ASC training will use class weights: {asc_weights_to_use}")
        else:
            logger.info("ASC training will not use class weights.")

        logger.info(f"Starting ASC training for {args.epochs} epochs...")
        _, _, eval_metrics = train_aspect_sentiment(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            num_epochs=args.epochs,
            output_dir=ASC_MODEL_PATH,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_clipping_norm=args.gradient_clipping,
            resume_from_checkpoint=args.resume,
            asc_class_weights=asc_weights_to_use
        )
        logger.info(f"ASC Training complete. Evaluation Macro F1: {eval_metrics.get('eval_macro_f1', 'N/A')}")

if __name__ == "__main__":
    main()

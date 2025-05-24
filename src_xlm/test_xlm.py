import json
import logging
import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
from seqeval.metrics import classification_report

from model import (
    # Constants
    ASPECT_MODEL_PATH, ASPECT_LABEL_MAP, ASPECT_LABEL_MAP_INVERSE, MODEL_NAME,
    
    # Classes and functions
    initialize_tokenizer, XLMRobertaForTokenClassificationCRF,
    recover_char_offsets, ABSAPipeline
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Test XLM-RoBERTa ATE model with CRF layer")
    parser.add_argument('--test_file', default='data/filtered_review_data_xlm/processed_aspect_data_test.json',
                        help='Path to test data file')
    parser.add_argument('--model_path', default=ASPECT_MODEL_PATH,
                        help='Path to trained model')
    parser.add_argument('--use_crf', action='store_true', default=True,
                        help='Use CRF layer for inference (default: True)')
    parser.add_argument('--output_file', default='results/xlm_predictions.json',
                        help='Path to save predictions')
    return parser.parse_args()

def evaluate_model(model, tokenizer, test_file, use_crf=True):
    """Evaluate the model on test data and compute metrics"""
    # Load test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Initialize lists to store true and predicted labels
    true_labels = []
    pred_labels = []
    
    # Initialize list to store predictions with character offsets
    predictions = []
    
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    # Evaluate each sample
    for item in tqdm(test_data, desc="Evaluating"):
        text = item['text']
        
        # Skip empty texts
        if not text:
            continue
        
        # Get ground truth BIO labels if available
        if 'bio_labels' in item:
            true_bio = item['bio_labels']
            true_labels.append(true_bio)
        
        # Tokenize the input
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_offsets_mapping=True
        )
        
        # Get offset mapping before moving tensors to device
        offset_mapping = encoding.pop('offset_mapping').squeeze().tolist()
        
        # Move to device
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        # Get predictions
        with torch.no_grad():
            if use_crf:
                # Use CRF model for prediction
                outputs = model(**encoding)
                if "predictions" in outputs:
                    predictions_tensor = outputs["predictions"].squeeze()
                else:
                    # Fall back to logits
                    logits = outputs["logits"]
                    predictions_tensor = torch.argmax(logits, dim=-1).squeeze()
            else:
                # Use standard model
                outputs = model(**encoding)
                predictions_tensor = torch.argmax(outputs.logits, dim=-1).squeeze()
        
        # Convert to BIO labels
        pred_bio = [ASPECT_LABEL_MAP_INVERSE[p.item()] for p in predictions_tensor]
        pred_labels.append(pred_bio)
        
        # Recover character-level offsets
        aspects = recover_char_offsets(
            tokenizer.convert_ids_to_tokens(encoding['input_ids'].squeeze().tolist()),
            predictions_tensor.cpu().numpy(),
            offset_mapping,
            text
        )
        
        # Save prediction with character offsets
        predictions.append({
            'text': text,
            'aspects': aspects
        })
    
    # Compute metrics
    if true_labels:
        # Trim prediction labels to match true labels length (tokenizer may produce different lengths)
        trimmed_pred_labels = []
        for true, pred in zip(true_labels, pred_labels):
            trimmed_pred_labels.append(pred[:len(true)])
        
        # Generate classification report
        report = classification_report(true_labels, trimmed_pred_labels, digits=4)
        logger.info(f"Classification report:\n{report}")
    
    return predictions

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = initialize_tokenizer()
    
    # Load model
    if args.use_crf:
        # Load the CRF model
        model = XLMRobertaForTokenClassificationCRF(MODEL_NAME, len(ASPECT_LABEL_MAP))
        # Load the saved state dict
        model.load_state_dict(torch.load(os.path.join(args.model_path, "pytorch_model.bin")))
    else:
        # Use standard HuggingFace model
        from transformers import AutoModelForTokenClassification
        model = AutoModelForTokenClassification.from_pretrained(args.model_path)
    
    # Evaluate model
    logger.info(f"Evaluating model from {args.model_path}")
    predictions = evaluate_model(model, tokenizer, args.test_file, args.use_crf)
    
    # Save predictions
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Predictions saved to {args.output_file}")
    
    # Test pipeline
    logger.info("Testing ABSAPipeline")
    pipeline = ABSAPipeline(aspect_model_path=args.model_path, use_crf=args.use_crf)
    
    # Test on a few examples
    example_texts = [
        "Το κινητό έχει καλή μπαταρία και η κάμερα βγάζει εξαιρετικές φωτογραφίες.",
        "Η οθόνη είναι μεγάλη αλλά η μπαταρία δεν αντέχει πολύ.",
        "Καλή ταχύτητα αλλά κακή κάμερα για φωτογραφίες."
    ]
    
    for i, text in enumerate(example_texts):
        logger.info(f"Example {i+1}: {text}")
        aspects = pipeline.extract_aspects(text)
        for aspect in aspects:
            logger.info(f"  Aspect: '{aspect['text']}' at positions {aspect['start']}-{aspect['end']}")

if __name__ == "__main__":
    main() 